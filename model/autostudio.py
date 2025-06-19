import os
import warnings
current_file_path = os.path.abspath(__file__)
dpath = os.path.dirname(current_file_path)
ppath = os.path.dirname(dpath)

import sys
sys.path.append(f"{ppath}/DETECT_SAM/")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Suppress common model loading warnings
warnings.filterwarnings("ignore", message=".*beta.*will be renamed internally to.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*will be renamed internally to.*weight.*")

from PIL import Image
from typing import List
from detectSam import process_image
from diffusers.pipelines.controlnet import MultiControlNetModel
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from model.utils import get_global_prompt, prepare_mid_image
from .utils import is_torch2_available, get_generator
import torchvision.transforms as transforms

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        AutoStudioAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, AutoStudioAttnProcessor
from .resampler import Resampler

def generate_layout_mask(fuse_scale, all_layouts, height=1024, width=1024):

    # generate layout mask
    n = len(all_layouts)
    
    fuse_scale = tuple(fuse_scale)
    if len(fuse_scale)!=2:
        fuse_scale = [1, 1]

    ID, LO, nson = [], [], 0
    for piece in all_layouts:
        ID.append(piece[0])
        LO.append(piece[1])
    all_layouts = LO

    # dim1=2，对应一个存prompt一个存ip
    # dim2=n+2，对应n个box和 一个带obj的背景、一个不带obj的背景
    layout = np.zeros((2, n + 2, height, width), dtype=np.float32)
    overlap_count = np.zeros((height, width), dtype=np.float32)
    for x, y, w, h in all_layouts:
        # overlap记录每个像素点重复在几个obj box中出现
        xi, yi, wi, hi = map(lambda v: int(round(v)), (x, y, w, h))
        overlap_count[yi:yi+hi, xi:xi+wi] += 1
    for i, (x, y, w, h) in enumerate(all_layouts):
        xi, yi, wi, hi = map(lambda v: int(round(v)), (x, y, w, h))

        # prompt的第零层对应带obj的背景层，1-fuse_scale[0]即背景的融合比例
        # prompt的第2层以后即obj层，0.7的fuse_scale，有两次重叠则每个obj分到0.35的权重
        layout[0, i+2, yi:yi+hi, xi:xi+wi] = 1 / overlap_count[yi:yi+hi, xi:xi+wi] * fuse_scale[0]
        layout[0, 0, yi:yi+hi, xi:xi+wi] = 0.3
        layout[1, i+2, yi:yi+hi, xi:xi+wi] = 1 / overlap_count[yi:yi+hi, xi:xi+wi]

    # prompt的第一层对应无obj的背景权重，用于补充权重未达到1的位置
    layout[0,1,:,:] = 1 - np.sum(layout[0,2:,:,:], axis=0) - layout[0,0,:,:]    # [cond; left; right])
    layout[1,1,:,:] = (1 - np.sum(layout[1,2:,:,:], axis=0)) * fuse_scale[1]

    return layout

def get_token_map(tokenizer, prompt, padding="do_not_pad"):
    fg_prompt_tokens = tokenizer([prompt], padding=padding, max_length=77, return_tensors="np")
    input_ids = fg_prompt_tokens['input_ids'][0]
    token_map = []
    for ind, item in enumerate(input_ids.tolist()):
        token = tokenizer._convert_id_to_token(item)
        token_map.append(token)
    return token_map

def scale_boxes_to_fit(boxes, target_size=512):

    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 0] + boxes[:, 2])
    y_max = np.max(boxes[:, 1] + boxes[:, 3])

    width = x_max - x_min
    height = y_max - y_min

    scale = target_size / max(width, height)
    scaled_boxes = (boxes - [x_min, y_min, 0, 0]) * scale

    new_width = width * scale
    new_height = height * scale
    x_offset = (target_size - new_width) / 2
    y_offset = (target_size - new_height) / 2

    scaled_boxes[:, 0] += x_offset
    scaled_boxes[:, 1] += y_offset

    return scaled_boxes, scale

def plot_scaled_boxes(key,boxes, scale, target_size=512):
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, target_size)
    ax.set_ylim(target_size, 0)  # 设置y轴的方向，使原点在左上角
    
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    bounding_rect = patches.Rectangle((0, 0), target_size, target_size,
                                      linewidth=2, edgecolor='b', facecolor='none', linestyle='--')
    ax.add_patch(bounding_rect)



def prepare_character(height, width, latent_scale, tokenizer, prompt_book, character_database):

    bboxes = []
    object_positions= []
    token_map = get_token_map(tokenizer,prompt_book['global_prompt'])
    token_map_str = " ".join(token_map)
    character_pb_list = []
    
    for obj in range(len(prompt_book['gen_boxes'])):
        #absolute_box = tuple([x / 512 for x in obj[1]])
        if str(prompt_book['obj_ids'][obj]).find("-") == -1:
            absolute_box = np.array([prompt_book['gen_boxes'][obj][1][0]/width ,prompt_book['gen_boxes'][obj][1][1]/height, prompt_book['gen_boxes'][obj][1][2]/width, prompt_book['gen_boxes'][obj][1][3]/height])
            bboxes.append(absolute_box)

    ref_imgs_now = [character_database['0'],character_database['0']] # No use image for Global and bg prompt
    for i in range(len(prompt_book['obj_ids'])):
        prompt_book['have_ref'][i] = 0
        try: 
            img = character_database[prompt_book['obj_ids'][i]]
            ref_imgs_now.append(img)
            prompt_book['have_ref'][i] = 1
        except:
            ref_imgs_now.append(character_database['0'])

    character_pb_dict = {}
    for i in range(len(prompt_book['obj_ids'])):
        obj_id = prompt_book['obj_ids'][i]
        if str(obj_id).find("-") == -1:
            # is mother id
            character_pb_dict[obj_id] = {
            "size": [width, height], #[height,width], fix big box for each character
            "gen_boxes": [prompt_book["gen_boxes"][i]],
            "bg_prompt": prompt_book['bg_prompt'],
            "global_prompt": '',
            "extra_neg_prompt": None,
            "obj_ids":[prompt_book["obj_ids"][i]],
            "have_ref":[prompt_book["have_ref"][i]],   
            "ref_imgs_now":[ref_imgs_now[0],ref_imgs_now[1],ref_imgs_now[i+2]]            
            }
            
    for i in range(len(prompt_book['obj_ids'])):
        obj_id = prompt_book['obj_ids'][i]
        if str(obj_id).find("-") == -1:
            continue            
        else:
            # is sub id
            character_pb_dict[obj_id.split("-")[0]]["obj_ids"].append(prompt_book["obj_ids"][i])
            character_pb_dict[obj_id.split("-")[0]]["have_ref"].append(prompt_book["have_ref"][i])
            character_pb_dict[obj_id.split("-")[0]]["gen_boxes"].append(prompt_book["gen_boxes"][i])
            character_pb_dict[obj_id.split("-")[0]]["ref_imgs_now"].append(ref_imgs_now[i+2])

    for key, pb in character_pb_dict.items():
        character_pb_dict[key] = get_global_prompt(character_pb_dict[key] )
        boxes = []
        #print(pb)
        for box in pb['gen_boxes']:
            boxes.append(list(box[1]))
            
        scaled_boxes, scale = scale_boxes_to_fit(np.array(boxes), target_size=1024)
        plot_scaled_boxes(key, scaled_boxes, scale, target_size=1024)      
        scaled_boxes = scaled_boxes.tolist()
        for i in range(len(scaled_boxes)):
            pb['gen_boxes'][i] = (pb['gen_boxes'][i][0], tuple(scaled_boxes[i]))

    return character_pb_dict, prompt_book, ref_imgs_now, bboxes, object_positions 

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class AUTOSTUDIO:
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        # Support MPS, CUDA, and CPU devices
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # load image encoder with appropriate dtype for device
        dtype = torch.float32 if self.device in ['cpu', 'mps'] else torch.float16
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=dtype
        )
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def encode_prompts(self, prompts):
        '''
        text_encoder output the hidden states
        prompts are based on list
        '''
        with torch.no_grad():
            tokens = self.pipe.tokenizer(prompts, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors='pt').input_ids.to(self.device)
            embs = self.pipe.text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(self.device, dtype = self.pipe.unet.dtype)
        return embs
    
    def init_proj(self):
        dtype = torch.float32 if self.device in ['cpu', 'mps'] else torch.float16
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=dtype)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                dtype = torch.float32 if self.device in ['cpu', 'mps'] else torch.float16
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            dtype = torch.float32 if self.device in ['cpu', 'mps'] else torch.float16
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=dtype)).image_embeds
        else:
            dtype = torch.float32 if self.device in ['cpu', 'mps'] else torch.float16
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        dino_model,
        same_model,
        character_database,
        prompt_book,
        do_latent_guidance,
        negative_prompt = None,
        img_scale = 1.0,
        num_samples = 4,
        seed = None,
        guidance_scale = 7.5,
        fuse_scale = [1, 0],
        num_inference_steps = 30,
        refine_step = 60,
        height = 1024,
        width = 1024,
        is_editing = False,
        repeat_ind = 0,
        scheduler='flow_match',  # Scheduler type for Flux
        max_sequence_length=512,  # Text encoder max length
        **kwargs,
    ):
        self.set_scale(img_scale)

        character_pb_dict, prompt_book, ref_imgs_now, bboxes, object_positions = prepare_character(height, width, self.pipe.vae_scale_factor, self.pipe.tokenizer, prompt_book, character_database)
        have_ref = prompt_book['have_ref']

        if do_latent_guidance:
            print('Generate Latent guidance')
            character_imgs_list = self.generate_latent_guidance(character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs)
            guide_imgs, guide_masks = [], []

            for i in character_imgs_list:
                character_prompt_full = ""
                for j in range(len(character_pb_dict[i[0]]['gen_boxes'])):
                    character_prompt = character_pb_dict[i[0]]['gen_boxes'][j][0]
                    character_prompt = character_prompt.split(",")[0]
                    character_prompt_full = character_prompt_full + character_prompt + ","
                character_prompt_full = character_prompt_full[:-1]

                seg_img, detection = process_image(detect_model=dino_model, same_model=same_model, input_image=i[1][0], categories=character_prompt_full, device=self.device)

                # add to ref_list if don't have ref now 
                sub_pb = character_pb_dict[i[0]]['obj_ids']
                for obj_id in range(len(sub_pb)):
                    ind = prompt_book['obj_ids'].index(sub_pb[obj_id])
                    
                    detected_id = list(detection.class_id)
                    try:
                        detecetd_position = detected_id.index(obj_id)
                        cropped_image = i[1][0].crop(detection.xyxy[detecetd_position])
                        ref_imgs_now[ind] = cropped_image
                        
                        if prompt_book['have_ref'][ind] == 0: # add to database
                            character_database[sub_pb[obj_id]] = cropped_image
                            prompt_book['have_ref'][ind] = 1

                        if obj_id == 0: # add to latent guidance (Mother ID only)
                            guide_masks.append(torch.tensor(detection.mask[detecetd_position]))
                            guide_imgs.append(i[1][0])

                    except:
                        continue

            # prepare latent_guidance
            latent_guidance_mask, latent_guidance_image = prepare_mid_image(guide_masks, guide_imgs, bboxes, height, width, repeat_ind)   
            latent_guidance_mask = latent_guidance_mask.resize((int(width/8), int(height/8)))
            latent_guidance_mask = np.array(latent_guidance_mask)
            latent_guidance_mask = torch.from_numpy(latent_guidance_mask).to(self.device)
            latent_guidance_mask[latent_guidance_mask < 255] = 0
            latent_guidance_mask = latent_guidance_mask.float() / 255.0
            latent_guidance_mask = 1.0 - latent_guidance_mask

            # update ref_imgs for global generation
            ref_imgs_now = [character_database['0'],character_database['0']] # 2 No use image for Global and Bg prompt
            for i in range(len(prompt_book['obj_ids'])):
                prompt_book['have_ref'][i] = 0
                try: 
                    img = character_database[prompt_book['obj_ids'][i]]
                    ref_imgs_now.append(img)
                    prompt_book['have_ref'][i] = 1
                except:
                    ref_imgs_now.append(character_database['0'])

        else:
            latent_guidance_mask = latent_guidance_image = None

        if prompt_book['global_prompt'] is None:
            all_prompts = ["best quality, high quality"]
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        else:
            negative_prompt = negative_prompt + "monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

            
        num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
        all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
        #print('Generate global prompt:',prompt_book['global_prompt'])
        all_layouts = []
                
        for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
            all_prompts.extend([p[0]])
            #all_layouts.append(p[1]) 
            all_layouts.append([obj_id, p[1]])
            

        if not isinstance(all_prompts, List):
            all_prompts = [all_prompts] * num_prompts

        layout_mask = generate_layout_mask(fuse_scale, all_layouts, height, width)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=ref_imgs_now, clip_image_embeds=None
        ) #[3,4,768]
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_ = self.encode_prompts(all_prompts) #[global,bg,cha1,cha2,...] #[4,77,768]
            negative_prompt_embeds_= self.encode_prompts([negative_prompt]) 
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        prompt_book_info = True

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            prompt_book_info=[have_ref, prompt_book_info],
            do_latent_guidance=do_latent_guidance,
            layout_mask = layout_mask,
            refine_step= refine_step,
            object_positions=object_positions,
            height = height,
            width = width,
            return_latents = False,
            mid_state_img = latent_guidance_image,
            mid_state_mask = latent_guidance_mask,
            is_editing = is_editing,
            repeat_ind=repeat_ind,
            **kwargs,
        ).images

        return [images, character_database]

    def generate_latent_guidance(self, character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs):

        character_imgs_list = []
        for character_id, prompt_book in character_pb_dict.items():
            ref_imgs_now = prompt_book['ref_imgs_now']
            have_ref = prompt_book['have_ref']
            if prompt_book['global_prompt'] is None:
                all_prompts = ["best quality, high quality"]
            if negative_prompt is None:
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            else:
                negative_prompt = negative_prompt + "monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

            num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
            all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
            #print('Generate global prompt:',prompt_book['global_prompt'])
            all_layouts = []
                    
            for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
                all_prompts.extend([p[0]])
                #all_layouts.append(p[1])
                all_layouts.append([obj_id, p[1]])
                

            if not isinstance(all_prompts, List):
                all_prompts = [all_prompts] * num_prompts

            layout_mask = generate_layout_mask(fuse_scale, all_layouts)

            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=ref_imgs_now, clip_image_embeds=None
            ) #[3,4,768]
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            with torch.inference_mode():
                prompt_embeds_ = self.encode_prompts(all_prompts) #[global,bg,cha1,cha2,...]
                negative_prompt_embeds_= self.encode_prompts([negative_prompt]) 
                prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)

            generator = get_generator(seed, self.device)

            prompt_book_info = True

            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=15,
                generator=generator,
                prompt_book_info=[have_ref, prompt_book_info],
                do_latent_guidance=False,
                layout_mask=layout_mask,
                refine_step=50,
                object_positions=[],
                height = 1024,
                width = 1024,
                return_latents = False,
                **kwargs,
            ).images
            character_imgs_list.append([character_id,images])

        return character_imgs_list

class AUTOSTUDIOPlus(AUTOSTUDIO):

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

class AUTOSTUDIOXL(AUTOSTUDIO):

    """AUTOSTUDIOXL"""

    def generate(
        self,
        dino_model,
        same_model,
        character_database,
        prompt_book,
        do_latent_guidance,
        negative_prompt = None,
        img_scale = 1.0,
        num_samples = 4,
        seed = None,
        guidance_scale = 7.5,
        fuse_scale = [1, 0],
        num_inference_steps = 30,
        refine_step = 60,
        height = 1024,
        width = 1024,
        is_editing = False,
        repeat_ind = 0,
        scheduler='flow_match',  # Scheduler type for Flux
        max_sequence_length=512,  # Text encoder max length
        **kwargs,
    ):

        self.set_scale(img_scale)

        character_pb_dict, prompt_book, ref_imgs_now, bboxes, object_positions = prepare_character(height, width, self.pipe.vae_scale_factor, self.pipe.tokenizer, prompt_book, character_database)
        have_ref = prompt_book['have_ref']

        if do_latent_guidance:
            print('Generate Latent guidance')
            character_imgs_list = self.generate_latent_guidanceXL(character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs)
            guide_imgs, guide_masks = [], []

            dino_model.model.to("cuda:1")
            same_model.to('cuda:1')

            for i in character_imgs_list:
                character_prompt_full = ""
                for j in range(len(character_pb_dict[i[0]]['gen_boxes'])):
                    character_prompt = character_pb_dict[i[0]]['gen_boxes'][j][0]
                    character_prompt = character_prompt.split(",")[0]
                    character_prompt_full = character_prompt_full + character_prompt + ","
                character_prompt_full = character_prompt_full[:-1]

                seg_img, detection = process_image(detect_model=dino_model, same_model=same_model, input_image=i[1][0], categories=character_prompt_full, device='cuda:1') #if CUDA out of spa

                # add to ref_list if don't have ref now 
                sub_pb = character_pb_dict[i[0]]['obj_ids']
                for obj_id in range(len(sub_pb)):
                    ind = prompt_book['obj_ids'].index(sub_pb[obj_id])
                    
                    detected_id = list(detection.class_id)
                    try:
                        detecetd_position = detected_id.index(obj_id)
                        cropped_image = i[1][0].crop(detection.xyxy[detecetd_position])
                        ref_imgs_now[ind] = cropped_image
                        
                        if prompt_book['have_ref'][ind] == 0: # add to database
                            character_database[sub_pb[obj_id]] = cropped_image
                            prompt_book['have_ref'][ind] = 1

                        if obj_id == 0: # add to latent guidance (Mother ID only)
                            guide_masks.append(torch.tensor(detection.mask[detecetd_position]))
                            guide_imgs.append(i[1][0])

                    except:
                        continue

            # prepare latent_guidance
            latent_guidance_mask, latent_guidance_image = prepare_mid_image(guide_masks, guide_imgs, bboxes, height, width, repeat_ind)   
            latent_guidance_mask = latent_guidance_mask.resize((int(width/8), int(height/8)))
            latent_guidance_mask = np.array(latent_guidance_mask)
            latent_guidance_mask = torch.from_numpy(latent_guidance_mask).to(self.device)
            latent_guidance_mask[latent_guidance_mask < 255] = 0
            latent_guidance_mask = latent_guidance_mask.float() / 255.0
            latent_guidance_mask = 1.0 - latent_guidance_mask

            # update ref_imgs for global generation
            ref_imgs_now = [character_database['0'],character_database['0']] # 2 No use image for Global and Bg prompt
            for i in range(len(prompt_book['obj_ids'])):
                prompt_book['have_ref'][i] = 0
                try: 
                    img = character_database[prompt_book['obj_ids'][i]]
                    ref_imgs_now.append(img)
                    prompt_book['have_ref'][i] = 1
                except:
                    ref_imgs_now.append(character_database['0'])
        else:
            latent_guidance_mask = latent_guidance_image = None

        if prompt_book['global_prompt'] is None:
            all_prompts = ["best quality, high quality"]
        if negative_prompt is None:
            negative_prompt = "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        else:
            negative_prompt = negative_prompt + "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            
        num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
        all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
        #print('Generate global prompt:',prompt_book['global_prompt'])
        all_layouts = []
                
        for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
            all_prompts.extend([p[0]])
            #all_layouts.append(p[1]) 
            all_layouts.append([obj_id, p[1]])
            

        if not isinstance(all_prompts, List):
            all_prompts = [all_prompts] * num_prompts

        layout_mask = generate_layout_mask(fuse_scale, all_layouts, height, width)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=ref_imgs_now, clip_image_embeds=None
        ) #[3,4,768]
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


        with torch.inference_mode():
            (
                prompt_embeds,
                _,
                _,
                _,
            ) = self.pipe.encode_prompt(
                all_prompts,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=None,
            )

            (
                _,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                [all_prompts[0]],
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt= [negative_prompt],
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
        prompt_book_info = True

        images = self.pipe(
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
            num_inference_steps = num_inference_steps,
            generator = generator,
            prompt_book_info = [have_ref, prompt_book_info],
            do_latent_guidance = do_latent_guidance,
            layout_mask = layout_mask,
            refine_step= refine_step,
            object_positions= object_positions,
            height = height,
            width = width,
            return_latents = False,
            mid_state_img = latent_guidance_image,
            mid_state_mask = latent_guidance_mask,
            is_editing = is_editing,
            repeat_ind=repeat_ind,
            **kwargs,
        ).images

        return [images, character_database]
    
    def generate_latent_guidanceXL(self, character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs):
        character_imgs_list = []
        for character_id, prompt_book in character_pb_dict.items():
            ref_imgs_now = prompt_book['ref_imgs_now']
            have_ref = prompt_book['have_ref']
            if prompt_book['global_prompt'] is None:
                all_prompts = ["best quality, high quality"]
            if negative_prompt is None:
                negative_prompt = "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            else:
                negative_prompt = negative_prompt + "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

                
            num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
            all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
            #print('Generate global prompt:',prompt_book['global_prompt'])
            all_layouts = []
                    
            for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
                all_prompts.extend([p[0]])
                #all_layouts.append(p[1])
                all_layouts.append([obj_id, p[1]])
                

            if not isinstance(all_prompts, List):
                all_prompts = [all_prompts] * num_prompts

            layout_mask = generate_layout_mask(fuse_scale, all_layouts)

            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=ref_imgs_now, clip_image_embeds=None
            ) #[3,4,768]
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            with torch.inference_mode():
                (
                    prompt_embeds,
                    _,
                    _,
                    _,
                ) = self.pipe.encode_prompt(
                    all_prompts,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=None,
                )

                (
                    _,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.pipe.encode_prompt(
                    [all_prompts[0]],
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt= [negative_prompt],
                )
                prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)
                generator = get_generator(seed, self.device)

            prompt_book_info = True

            images = self.pipe(
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = pooled_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                num_inference_steps = 10,
                generator = generator,
                prompt_book_info = [have_ref, prompt_book_info],
                do_latent_guidance = False,
                layout_mask = layout_mask,
                refine_step= 50,
                object_positions= [],
                height = 1024,
                width = 1024,
                return_latents = False,
                **kwargs,
            ).images

            character_imgs_list.append([character_id, images])
        return character_imgs_list

# Flux.1 Classes
class AUTOSTUDIOFLUX:
    def __init__(self, flux_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        # Support MPS, CUDA, and CPU devices
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        print(f'Moving Flux pipeline to device: {self.device}')
        
        # For MPS, avoid moving large models to device immediately due to memory constraints
        if self.device == 'mps':
            print('Keeping Flux pipeline on original device for MPS compatibility')
            self.pipe = flux_pipe  # Don't move to MPS yet
        else:
            self.pipe = flux_pipe.to(self.device)
        
        print('Loading image encoder...')
        # load image encoder with appropriate dtype for device
        if self.device in ['cpu', 'mps']:
            dtype = torch.float16  # Use float16 for MPS memory efficiency
        else:  # CUDA
            dtype = torch.bfloat16
        
        try:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=dtype
            )
            print(f'Image encoder loaded successfully on {self.device} with dtype {dtype}')
        except Exception as e:
            print(f'Error loading image encoder on {self.device}: {e}')
            # Fallback to CPU for image encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                'cpu', dtype=torch.float32
            )
            print('Image encoder loaded on CPU as fallback')
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model for Flux
        self.image_proj_model = self.init_proj()

    def encode_prompts(self, prompts):
        '''
        For Flux, we need to handle both CLIP and T5 encoders
        '''
        with torch.no_grad():
            # Using Flux's encode_prompt method
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompts if isinstance(prompts, list) else [prompts],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        return prompt_embeds
    
    def init_proj(self):
        # Adapted for Flux's different embedding dimensions
        image_proj_model = ImageProjModel(
            cross_attention_dim=3072,  # Flux transformer dimension
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.bfloat16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        # For Flux, we'll need a different scaling approach
        pass
    
    def configure_scheduler(self, scheduler_type='flow_match'):
        """Configure scheduler for Flux pipeline"""
        try:
            from diffusers.schedulers import (
                FlowMatchEulerDiscreteScheduler,
                DDIMScheduler,
                EulerDiscreteScheduler,
                DPMSolverMultistepScheduler
            )
            
            if scheduler_type == 'flow_match':
                # Default Flux scheduler
                if hasattr(self.pipe, 'scheduler') and isinstance(self.pipe.scheduler, FlowMatchEulerDiscreteScheduler):
                    return  # Already using the right scheduler
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'euler':
                scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'ddim':
                scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'dpm':
                scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            else:
                print(f"⚠️ Unknown scheduler '{scheduler_type}', using default flow_match")
                return
                
            self.pipe.scheduler = scheduler
            print(f"✅ Configured scheduler: {scheduler_type}")
            
        except Exception as e:
            print(f"⚠️ Failed to configure scheduler '{scheduler_type}': {e}")
            print("Using default scheduler")

    def generate(
        self,
        dino_model,
        same_model,
        character_database,
        prompt_book,
        do_latent_guidance,
        negative_prompt=None,
        img_scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=3.5,  # Lower default for Flux
        fuse_scale=[1, 0],
        num_inference_steps=4,  # Much fewer steps for Flux
        refine_step=4,
        height=1024,
        width=1024,
        is_editing=False,
        repeat_ind=0,
        scheduler='flow_match',  # Scheduler type for Flux
        max_sequence_length=512,  # Text encoder max length
        **kwargs,
    ):
        
        # Configure scheduler
        self.configure_scheduler(scheduler)
        
        # Character consistency for Flux - build comprehensive prompt with character references
        character_descriptions = []
        character_refs = []
        
        # Process characters for consistency
        for i, obj_id in enumerate(prompt_book['obj_ids']):
            char_desc = prompt_book['gen_boxes'][i][0]  # Character description
            character_descriptions.append(f"({char_desc})")
            
            # Check if we have a reference image for this character
            if obj_id in character_database and character_database[obj_id] != character_database.get('0'):
                character_refs.append(f"consistent with previous appearance")
            else:
                character_refs.append(f"new character introduction")
        
        # Build comprehensive prompt with character consistency cues
        if character_descriptions:
            character_prompt = ", ".join(character_descriptions)
            consistency_cues = " ".join(character_refs)
            bg_prompt = prompt_book.get('bg_prompt', prompt_book.get('background', ''))
            scene_prompt = prompt_book.get('prompt', prompt_book.get('caption', ''))
            main_prompt = f"{bg_prompt}, {character_prompt}, {consistency_cues}, {scene_prompt}"
        else:
            main_prompt = prompt_book['global_prompt'] if prompt_book['global_prompt'] else "best quality, high quality"
        
        print(f"🎭 Character-aware prompt: {main_prompt[:100]}...")
        print(f"📊 Processing {len(character_descriptions)} characters")
        
        # Handle MPS tensor allocation issues with sophisticated workaround
        if self.device == 'mps':
            print("Attempting MPS-native generation with tensor allocation workarounds...")
            
            # Try multiple strategies for MPS generation
            images = None
            strategies = [
                "mps_with_cpu_generator",
                "mps_with_pre_allocation", 
                "fallback_to_cpu"
            ]
            
            for strategy in strategies:
                try:
                    if strategy == "mps_with_cpu_generator":
                        print("Strategy 1: MPS generation with CPU generator...")
                        generator = get_generator(seed, 'cpu')
                        
                        # Ensure pipeline is on MPS
                        if next(self.pipe.parameters()).device != torch.device('mps'):
                            self.pipe = self.pipe.to('mps')
                        
                        # Pre-allocate MPS memory
                        torch.mps.empty_cache()
                        
                        images = self.pipe(
                            prompt=main_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            max_sequence_length=max_sequence_length,
                            **kwargs,
                        ).images
                        print("✅ MPS generation successful with CPU generator!")
                        break
                        
                    elif strategy == "mps_with_pre_allocation":
                        print("Strategy 2: MPS generation with tensor pre-allocation...")
                        generator = get_generator(seed, self.device)
                        
                        # Pre-allocate tensors to avoid runtime allocation issues
                        batch_size = 1
                        channels = 16  # Flux latent channels
                        latent_height = height // 8
                        latent_width = width // 8
                        
                        # Pre-allocate latent space on MPS
                        dummy_latents = torch.randn(
                            batch_size, channels, latent_height, latent_width,
                            dtype=torch.float16, device='cpu'
                        ).to('mps')
                        print(f"Pre-allocated latents: {dummy_latents.shape} on {dummy_latents.device}")
                        
                        # Clear the dummy tensor
                        del dummy_latents
                        torch.mps.empty_cache()
                        
                        images = self.pipe(
                            prompt=main_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            max_sequence_length=max_sequence_length,
                            **kwargs,
                        ).images
                        print("✅ MPS generation successful with pre-allocation!")
                        break
                        
                except Exception as e:
                    print(f"❌ {strategy} failed: {e}")
                    if strategy == "fallback_to_cpu":
                        raise e
                    continue
            
            # Final fallback to CPU if all MPS strategies fail
            if images is None:
                print("🔄 All MPS strategies failed, falling back to CPU...")
                generator = get_generator(seed, 'cpu')
                self.pipe = self.pipe.to('cpu')
                
                images = self.pipe(
                    prompt=main_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs,
                ).images
                print("Generation completed on CPU (fallback)")
        else:
            generator = get_generator(seed, self.device)
            images = self.pipe(
                prompt=main_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            ).images

        # Update character database with generated character crops for future consistency
        if images and len(images) > 0:
            generated_image = images[0]
            
            # Extract character crops based on bounding boxes
            for i, obj_id in enumerate(prompt_book['obj_ids']):
                try:
                    # Get bounding box coordinates
                    bbox = prompt_book['gen_boxes'][i][1]  # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # Ensure coordinates are within image bounds
                    img_width, img_height = generated_image.size
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))
                    
                    # Crop character from generated image
                    character_crop = generated_image.crop((x, y, x + w, y + h))
                    
                    # Store in character database for future consistency
                    character_database[obj_id] = character_crop
                    print(f"📸 Stored character {obj_id} crop ({w}x{h}) for future consistency")
                    
                except Exception as e:
                    print(f"⚠️ Failed to extract character {obj_id}: {e}")
                    continue

        return [images, character_database]

class AUTOSTUDIOFLUXPlus(AUTOSTUDIOFLUX):
    def init_proj(self):
        # Use Resampler for better image understanding in Flux
        image_proj_model = Resampler(
            dim=3072,  # Flux transformer dimension
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=3072,
            ff_mult=4,
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.bfloat16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

class AUTOSTUDIOXLPlus(AUTOSTUDIO):
    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        dino_model,
        same_model,
        character_database,
        prompt_book,
        do_latent_guidance,
        negative_prompt = None,
        img_scale = 1.0,
        num_samples = 4,
        seed = None,
        guidance_scale = 7.5,
        fuse_scale = [1, 0],
        num_inference_steps = 30,
        refine_step = 60,
        height = 1024,
        width = 1024,
        is_editing = False,
        repeat_ind = 0,
        scheduler='flow_match',  # Scheduler type for Flux
        max_sequence_length=512,  # Text encoder max length
        **kwargs,
    ):

        self.set_scale(img_scale)

        character_pb_dict, prompt_book, ref_imgs_now, bboxes, object_positions = prepare_character(height, width, self.pipe.vae_scale_factor, self.pipe.tokenizer, prompt_book, character_database)
        have_ref = prompt_book['have_ref']

        if do_latent_guidance:
            print('Generate Latent guidance')
            character_imgs_list = self.generate_latent_guidanceXL(character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs)
            guide_imgs, guide_masks = [], []

            dino_model.model.to("cuda:1")
            same_model.to('cuda:1')

            for i in character_imgs_list:
                character_prompt_full = ""
                for j in range(len(character_pb_dict[i[0]]['gen_boxes'])):
                    character_prompt = character_pb_dict[i[0]]['gen_boxes'][j][0]
                    character_prompt = character_prompt.split(",")[0]
                    character_prompt_full = character_prompt_full + character_prompt + ","
                character_prompt_full = character_prompt_full[:-1]

                seg_img, detection = process_image(detect_model=dino_model, same_model=same_model, input_image=i[1][0], categories=character_prompt_full, device='cuda:1') #if CUDA out of spa

                # add to ref_list if don't have ref now 
                sub_pb = character_pb_dict[i[0]]['obj_ids']
                for obj_id in range(len(sub_pb)):
                    ind = prompt_book['obj_ids'].index(sub_pb[obj_id])
                    
                    detected_id = list(detection.class_id)
                    try:
                        detecetd_position = detected_id.index(obj_id)
                        cropped_image = i[1][0].crop(detection.xyxy[detecetd_position])
                        ref_imgs_now[ind] = cropped_image
                        
                        if prompt_book['have_ref'][ind] == 0: # add to database
                            character_database[sub_pb[obj_id]] = cropped_image
                            prompt_book['have_ref'][ind] = 1

                        if obj_id == 0: # add to latent guidance (Mother ID only)
                            guide_masks.append(torch.tensor(detection.mask[detecetd_position]))
                            guide_imgs.append(i[1][0])

                    except:
                        continue

            # prepare latent_guidance
            latent_guidance_mask, latent_guidance_image = prepare_mid_image(guide_masks, guide_imgs, bboxes, height, width, repeat_ind)  
            latent_guidance_mask = latent_guidance_mask.resize((int(width/8), int(height/8)))
            latent_guidance_mask = np.array(latent_guidance_mask)
            latent_guidance_mask = torch.from_numpy(latent_guidance_mask).to(self.device)
            latent_guidance_mask[latent_guidance_mask < 255] = 0
            latent_guidance_mask = latent_guidance_mask.float() / 255.0
            latent_guidance_mask = 1.0 - latent_guidance_mask

            # update ref_imgs for global generation
            ref_imgs_now = [character_database['0'],character_database['0']] # 2 No use image for Global and Bg prompt
            for i in range(len(prompt_book['obj_ids'])):
                prompt_book['have_ref'][i] = 0
                try: 
                    img = character_database[prompt_book['obj_ids'][i]]
                    ref_imgs_now.append(img)
                    prompt_book['have_ref'][i] = 1
                except:
                    ref_imgs_now.append(character_database['0'])
        else:
            latent_guidance_mask = latent_guidance_image = None

        if prompt_book['global_prompt'] is None:
            all_prompts = ["best quality, high quality"]
        
        if negative_prompt is None:
            negative_prompt = "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        else:
            negative_prompt = negative_prompt + "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            
        num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
        all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
        #print('Generate global prompt:',prompt_book['global_prompt'])
        all_layouts = []
                
        for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
            all_prompts.extend([p[0]])
            #all_layouts.append(p[1]) 
            all_layouts.append([obj_id, p[1]])
            

        if not isinstance(all_prompts, List):
            all_prompts = [all_prompts] * num_prompts

        layout_mask = generate_layout_mask(fuse_scale, all_layouts, height, width)

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=ref_imgs_now) #[3,4,768]
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)


        with torch.inference_mode():
            (
                prompt_embeds,
                _,
                _,
                _,
            ) = self.pipe.encode_prompt(
                all_prompts,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=None,
            )

            (
                _,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                [all_prompts[0]],
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt= [negative_prompt],
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
        prompt_book_info = True

        images = self.pipe(
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
            num_inference_steps = num_inference_steps,
            generator = generator,
            prompt_book_info = [have_ref, prompt_book_info],
            do_latent_guidance = do_latent_guidance,
            layout_mask = layout_mask,
            refine_step= refine_step,
            object_positions= object_positions,
            height = height,
            width = width,
            return_latents = False,
            mid_state_img = latent_guidance_image,
            mid_state_mask = latent_guidance_mask,
            is_editing = is_editing,
            **kwargs,
        ).images

        return [images, character_database]
    
    def generate_latent_guidanceXL(self, character_pb_dict, negative_prompt, img_scale, num_samples, seed, guidance_scale, fuse_scale, num_inference_steps, height, width, **kwargs):
        character_imgs_list = []
        for character_id, prompt_book in character_pb_dict.items():
            ref_imgs_now = prompt_book['ref_imgs_now']
            have_ref = prompt_book['have_ref']
            if prompt_book['global_prompt'] is None:
                all_prompts = ["best quality, high quality"]
            if negative_prompt is None:
                negative_prompt = "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            else:
                negative_prompt = negative_prompt + "comic, cartoon, monochrome, lowres, bad anatomy, worst quality, low quality, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, pgly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
                
            num_prompts = 1 if isinstance(ref_imgs_now, Image.Image) else len(ref_imgs_now)
            all_prompts = [prompt_book['global_prompt'],prompt_book['bg_prompt']]
            #print('Generate global prompt:',prompt_book['global_prompt'])
            all_layouts = []
                    
            for p, obj_id in zip(prompt_book['gen_boxes'],prompt_book['obj_ids']):
                all_prompts.extend([p[0]])
                #all_layouts.append(p[1])
                all_layouts.append([obj_id, p[1]])
                

            if not isinstance(all_prompts, List):
                all_prompts = [all_prompts] * num_prompts

            layout_mask = generate_layout_mask(fuse_scale, all_layouts)

            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=ref_imgs_now,) #[3,4,768]
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            with torch.inference_mode():
                (
                    prompt_embeds,
                    _,
                    _,
                    _,
                ) = self.pipe.encode_prompt(
                    all_prompts,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=None,
                )

                (
                    _,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.pipe.encode_prompt(
                    [all_prompts[0]],
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt= [negative_prompt],
                )
                prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds.repeat(prompt_embeds.size()[0], 1, 1), uncond_image_prompt_embeds], dim=1)
                generator = get_generator(seed, self.device)

            prompt_book_info = True

            images = self.pipe(
                prompt_embeds = prompt_embeds,
                negative_prompt_embeds = negative_prompt_embeds,
                pooled_prompt_embeds = pooled_prompt_embeds,
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds,
                num_inference_steps = 10,
                generator = generator,
                prompt_book_info = [have_ref, prompt_book_info],
                do_latent_guidance = False,
                layout_mask = layout_mask,
                refine_step= 50,
                object_positions= [],
                height = 1024,
                width = 1024,
                return_latents = False,
                **kwargs,
            ).images

            character_imgs_list.append([character_id, images])
        return character_imgs_list

# Flux.1 Classes
class AUTOSTUDIOFLUX:
    def __init__(self, flux_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        # Support MPS, CUDA, and CPU devices
        if device == 'mps' and torch.backends.mps.is_available():
            self.device = 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        print(f'Moving Flux pipeline to device: {self.device}')
        
        # For MPS, avoid moving large models to device immediately due to memory constraints
        if self.device == 'mps':
            print('Keeping Flux pipeline on original device for MPS compatibility')
            self.pipe = flux_pipe  # Don't move to MPS yet
        else:
            self.pipe = flux_pipe.to(self.device)
        
        print('Loading image encoder...')
        # load image encoder with appropriate dtype for device
        if self.device in ['cpu', 'mps']:
            dtype = torch.float16  # Use float16 for MPS memory efficiency
        else:  # CUDA
            dtype = torch.bfloat16
        
        try:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=dtype
            )
            print(f'Image encoder loaded successfully on {self.device} with dtype {dtype}')
        except Exception as e:
            print(f'Error loading image encoder on {self.device}: {e}')
            # Fallback to CPU for image encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                'cpu', dtype=torch.float32
            )
            print('Image encoder loaded on CPU as fallback')
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model for Flux
        self.image_proj_model = self.init_proj()

    def encode_prompts(self, prompts):
        '''
        For Flux, we need to handle both CLIP and T5 encoders
        '''
        with torch.no_grad():
            # Using Flux's encode_prompt method
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompts if isinstance(prompts, list) else [prompts],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        return prompt_embeds
    
    def init_proj(self):
        # Adapted for Flux's different embedding dimensions
        image_proj_model = ImageProjModel(
            cross_attention_dim=3072,  # Flux transformer dimension
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.bfloat16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        # For Flux, we'll need a different scaling approach
        pass
    
    def configure_scheduler(self, scheduler_type='flow_match'):
        """Configure scheduler for Flux pipeline"""
        try:
            from diffusers.schedulers import (
                FlowMatchEulerDiscreteScheduler,
                DDIMScheduler,
                EulerDiscreteScheduler,
                DPMSolverMultistepScheduler
            )
            
            if scheduler_type == 'flow_match':
                # Default Flux scheduler
                if hasattr(self.pipe, 'scheduler') and isinstance(self.pipe.scheduler, FlowMatchEulerDiscreteScheduler):
                    return  # Already using the right scheduler
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'euler':
                scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'ddim':
                scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            elif scheduler_type == 'dpm':
                scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            else:
                print(f"⚠️ Unknown scheduler '{scheduler_type}', using default flow_match")
                return
                
            self.pipe.scheduler = scheduler
            print(f"✅ Configured scheduler: {scheduler_type}")
            
        except Exception as e:
            print(f"⚠️ Failed to configure scheduler '{scheduler_type}': {e}")
            print("Using default scheduler")

    def generate(
        self,
        dino_model,
        same_model,
        character_database,
        prompt_book,
        do_latent_guidance,
        negative_prompt=None,
        img_scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=3.5,  # Lower default for Flux
        fuse_scale=[1, 0],
        num_inference_steps=4,  # Much fewer steps for Flux
        refine_step=4,
        height=1024,
        width=1024,
        is_editing=False,
        repeat_ind=0,
        scheduler='flow_match',  # Scheduler type for Flux
        max_sequence_length=512,  # Text encoder max length
        **kwargs,
    ):
        
        # Configure scheduler
        self.configure_scheduler(scheduler)
        
        # Character consistency for Flux - build comprehensive prompt with character references
        character_descriptions = []
        character_refs = []
        
        # Process characters for consistency
        for i, obj_id in enumerate(prompt_book['obj_ids']):
            char_desc = prompt_book['gen_boxes'][i][0]  # Character description
            character_descriptions.append(f"({char_desc})")
            
            # Check if we have a reference image for this character
            if obj_id in character_database and character_database[obj_id] != character_database.get('0'):
                character_refs.append(f"consistent with previous appearance")
            else:
                character_refs.append(f"new character introduction")
        
        # Build comprehensive prompt with character consistency cues
        if character_descriptions:
            character_prompt = ", ".join(character_descriptions)
            consistency_cues = " ".join(character_refs)
            bg_prompt = prompt_book.get('bg_prompt', prompt_book.get('background', ''))
            scene_prompt = prompt_book.get('prompt', prompt_book.get('caption', ''))
            main_prompt = f"{bg_prompt}, {character_prompt}, {consistency_cues}, {scene_prompt}"
        else:
            main_prompt = prompt_book['global_prompt'] if prompt_book['global_prompt'] else "best quality, high quality"
        
        print(f"🎭 Character-aware prompt: {main_prompt[:100]}...")
        print(f"📊 Processing {len(character_descriptions)} characters")
        
        # Handle MPS tensor allocation issues with sophisticated workaround
        if self.device == 'mps':
            print("Attempting MPS-native generation with tensor allocation workarounds...")
            
            # Try multiple strategies for MPS generation
            images = None
            strategies = [
                "mps_with_cpu_generator",
                "mps_with_pre_allocation", 
                "fallback_to_cpu"
            ]
            
            for strategy in strategies:
                try:
                    if strategy == "mps_with_cpu_generator":
                        print("Strategy 1: MPS generation with CPU generator...")
                        generator = get_generator(seed, 'cpu')
                        
                        # Ensure pipeline is on MPS
                        if next(self.pipe.parameters()).device != torch.device('mps'):
                            self.pipe = self.pipe.to('mps')
                        
                        # Pre-allocate MPS memory
                        torch.mps.empty_cache()
                        
                        images = self.pipe(
                            prompt=main_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            max_sequence_length=max_sequence_length,
                            **kwargs,
                        ).images
                        print("✅ MPS generation successful with CPU generator!")
                        break
                        
                    elif strategy == "mps_with_pre_allocation":
                        print("Strategy 2: MPS generation with tensor pre-allocation...")
                        generator = get_generator(seed, self.device)
                        
                        # Pre-allocate tensors to avoid runtime allocation issues
                        batch_size = 1
                        channels = 16  # Flux latent channels
                        latent_height = height // 8
                        latent_width = width // 8
                        
                        # Pre-allocate latent space on MPS
                        dummy_latents = torch.randn(
                            batch_size, channels, latent_height, latent_width,
                            dtype=torch.float16, device='cpu'
                        ).to('mps')
                        print(f"Pre-allocated latents: {dummy_latents.shape} on {dummy_latents.device}")
                        
                        # Clear the dummy tensor
                        del dummy_latents
                        torch.mps.empty_cache()
                        
                        images = self.pipe(
                            prompt=main_prompt,
                            height=height,
                            width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            max_sequence_length=max_sequence_length,
                            **kwargs,
                        ).images
                        print("✅ MPS generation successful with pre-allocation!")
                        break
                        
                except Exception as e:
                    print(f"❌ {strategy} failed: {e}")
                    if strategy == "fallback_to_cpu":
                        raise e
                    continue
            
            # Final fallback to CPU if all MPS strategies fail
            if images is None:
                print("🔄 All MPS strategies failed, falling back to CPU...")
                generator = get_generator(seed, 'cpu')
                self.pipe = self.pipe.to('cpu')
                
                images = self.pipe(
                    prompt=main_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **kwargs,
                ).images
                print("Generation completed on CPU (fallback)")
        else:
            generator = get_generator(seed, self.device)
            images = self.pipe(
                prompt=main_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            ).images

        # Update character database with generated character crops for future consistency
        if images and len(images) > 0:
            generated_image = images[0]
            
            # Extract character crops based on bounding boxes
            for i, obj_id in enumerate(prompt_book['obj_ids']):
                try:
                    # Get bounding box coordinates
                    bbox = prompt_book['gen_boxes'][i][1]  # [x, y, w, h]
                    x, y, w, h = bbox
                    
                    # Ensure coordinates are within image bounds
                    img_width, img_height = generated_image.size
                    x = max(0, min(x, img_width - 1))
                    y = max(0, min(y, img_height - 1))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))
                    
                    # Crop character from generated image
                    character_crop = generated_image.crop((x, y, x + w, y + h))
                    
                    # Store in character database for future consistency
                    character_database[obj_id] = character_crop
                    print(f"📸 Stored character {obj_id} crop ({w}x{h}) for future consistency")
                    
                except Exception as e:
                    print(f"⚠️ Failed to extract character {obj_id}: {e}")
                    continue

        return [images, character_database]

class AUTOSTUDIOFLUXPlus(AUTOSTUDIOFLUX):
    def init_proj(self):
        # Use Resampler for better image understanding in Flux
        image_proj_model = Resampler(
            dim=3072,  # Flux transformer dimension
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=3072,
            ff_mult=4,
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.bfloat16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
