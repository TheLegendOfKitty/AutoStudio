'''

Created by Junhao Cheng 
22/03/2024

'''

from logging import config
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
current_file_path = os.path.abspath(__file__)
dpath = os.path.dirname(current_file_path)

import sys
sys.path.append(f"{dpath}/DETECT_SAM/")

import argparse
import numpy as np
import torch
import json
import time
import os
from accelerate import Accelerator

# Enable MPS fallback for unsupported operations
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Set MPS memory management environment variables before any PyTorch operations
if torch.backends.mps.is_available():
    # Allow MPS to use more memory when needed
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # Enable memory fraction control
    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

from PIL import Image
from safetensors.torch import load_file
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import BitsAndBytesConfig
# Import SD models only when needed
if len(sys.argv) > 1 and '--sd_version' in sys.argv:
    sd_version_idx = sys.argv.index('--sd_version') + 1
    if sd_version_idx < len(sys.argv) and sys.argv[sd_version_idx] in ['flux', 'fluxplus']:
        # For Flux, only import Flux-related modules
        from diffusers import FluxPipeline
        from model.utils import show_boxes, show_image, get_global_prompt
        from model.autostudio import AUTOSTUDIOFLUX, AUTOSTUDIOFLUXPlus
    else:
        # For SD models, import SD-related modules
        from model.pipeline_stable_diffusion import StableDiffusionPipeline
        from model.pipeline_stable_diffusionxl import StableDiffusionXLPipeline
        from model.unet_2d_condition import UNet2DConditionModel
        from model.utils import show_boxes, show_image, get_global_prompt
        from model.autostudio import AUTOSTUDIO, AUTOSTUDIOPlus, AUTOSTUDIOXL, AUTOSTUDIOXLPlus
else:
    # Default imports for backward compatibility
    from model.pipeline_stable_diffusion import StableDiffusionPipeline
    from model.pipeline_stable_diffusionxl import StableDiffusionXLPipeline
    from model.unet_2d_condition import UNet2DConditionModel
    from model.utils import show_boxes, show_image, get_global_prompt
    from model.autostudio import AUTOSTUDIO, AUTOSTUDIOPlus, AUTOSTUDIOXL, AUTOSTUDIOXLPlus

from detectSam import EFFICIENT_SAM_MODEL, GROUNDING_DINO_MODEL


LARGE_CONSTANT = 520
LARGE_CONSTANT2 = 5201314

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='cache/demo.json', type=str, help="Dialogue instruction data path")
parser.add_argument("--sd_path", default='/data2/chengjunhao/THEATERGEN/pretrained_models/diffusion_1.5_comic', type=str, help="Path to Stable Diffusion Folder")
parser.add_argument("--vae_path", default='/data2/chengjunhao/THEATERGEN/pretrained_models/vae_ft_mse', type=str, help="Path to VAE Folder")
parser.add_argument("--repeats", default=2, type=int, help="Number of samples for each prompt")
parser.add_argument("--seed_offset", default=1, type=int, help="Offset to the seed (seed starts from this number)")
parser.add_argument("--sd_version", default='1.5plus', type=str, help="Base model version. Pick from [1.5, 1.5plus, xl, xlplus, flux, fluxplus]") # 1.5, 1.5plus is recommended
parser.add_argument("--save_dir", default='output', type=str, help="Database path")
parser.add_argument("--freeze_dialogue_seed", default = 1, type=int, help="Use same seed for each dialogue for more consistency")
parser.add_argument("--do_latent_guidance", default = True, type=bool, help="Latent initialization for each character")
parser.add_argument("--is_editing", default = False, type=bool, help="Multi-turn editing mode")
parser.add_argument("--is_CMIGBENCH", default = False, type=bool, help="Multi-turn editing mode")
parser.add_argument("--device", default ='auto', type=str, help="Run Device (auto/cuda/mps/cpu)")
parser.add_argument("--quantize_8bit", action='store_true', help="Enable 8-bit quantization to reduce memory usage")
parser.add_argument("--quantization", type=str, choices=['Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q3_K_M'], help="GGUF quantization level for Flux models")
parser.add_argument("--gguf_path", type=str, help="Path to pre-downloaded GGUF model file")
parser.add_argument("--model_variant", default='schnell', choices=['schnell', 'dev'], help="Flux model variant for GGUF")


args = parser.parse_args()

# Initialize Accelerator for optimal device selection
accelerator = Accelerator()

# Auto-detect best device if not specified
if args.device == 'auto':
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        print('Using MPS (Metal Performance Shaders) for acceleration')
    elif torch.cuda.is_available():
        device = 'cuda'
        print('Using CUDA for acceleration')
    else:
        device = 'cpu'
        print('Using CPU (no GPU acceleration available)')
else:
    device = args.device
    print(f'Using specified device: {device}')

# Validate MPS availability if explicitly requested
if device == 'mps' and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
    print('Warning: MPS not available, falling back to CPU')
    device = 'cpu'
do_latent_guidance = args.do_latent_guidance
is_editing = args.is_editing # will be released soon
is_CMIGBENCH = args.is_CMIGBENCH

print('Welcome to the AutoStudio')

if args.sd_version == '1.5':
    ip_ckpt = "/IP-Adapter/models/ip-adapter_sd15.bin"
    image_encoder_path = "/IP-Adapter/models/image_encoder/"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )   
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=torch.float16)

    '''
    recommend comic style checkpoints:
    '''
    unet = UNet2DConditionModel.from_pretrained('diffusion_1.5/unet').to(dtype=torch.float16) 
    
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        unet=unet,
        feature_extractor=None,
        safety_checker=None
    )

    autostudio = AUTOSTUDIO(sd_pipe, image_encoder_path, ip_ckpt, device)
    print('Succesfully load models')

elif args.sd_version == '1.5plus':
    ip_ckpt = "/IP-Adapter/models/ip-adapter-plus_sd15.bin"
    image_encoder_path = "/IP-Adapter/models/image_encoder/"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )   
    
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained('/diffusion_1.5/unet').to(dtype=torch.float16)
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        unet=unet,
        feature_extractor=None,
        safety_checker=None
    )

    autostudio = AUTOSTUDIOPlus(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    print('Succesfully load models')

elif args.sd_version == 'xl':
    base_model_path = "pretrained_models/diffusion_xl"
    image_encoder_path = "pretrained_models/image_adapter/sdxl_models/image_encoder/"
    ip_ckpt = "pretrained_models/image_adapter/sdxl_models/ip-adapter_sdxl.bin"
    unet = UNet2DConditionModel.from_pretrained('pretrained_models/diffusion_xl/unet').to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae').to(dtype=torch.float16)

    sd_pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        unet = unet,
        vae = vae,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    autostudio = AUTOSTUDIOXL(sd_pipe, image_encoder_path, ip_ckpt, device)

elif args.sd_version == 'xlplus':
    base_model_path = "pretrained_models/diffusion_xl"
    image_encoder_path = "/IP-Adapter/models/image_encoder"
    ip_ckpt = "/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
    unet = UNet2DConditionModel.from_pretrained('pretrained_models/diffusion_xl/unet').to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae').to(dtype=torch.float16)

    sd_pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        unet = unet,
        vae = vae,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    autostudio = AUTOSTUDIOXLPlus(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

elif args.sd_version == 'flux':
    image_encoder_path = "openai/clip-vit-base-patch32"
    ip_ckpt = "./models/dummy_ip.bin"  # Placeholder for now
    
    # Check if GGUF quantization is requested
    if args.quantization or args.gguf_path:
        print('🎯 Loading Flux with GGUF quantization...')
        from flux_quantization import FluxGGUFManager, get_memory_info
        
        manager = FluxGGUFManager()
        
        # Auto-recommend quantization if not specified
        if not args.quantization and not args.gguf_path:
            memory_gb = get_memory_info(device)
            args.quantization = manager.recommend_quantization(device, memory_gb)
            print(f"💡 Auto-selected quantization: {args.quantization}")
        
        # Create quantized pipeline
        flux_pipe = manager.create_quantized_pipeline(
            model_variant=args.model_variant,
            quantization=args.quantization,
            device=device,
            gguf_path=args.gguf_path
        )
        
        if not flux_pipe:
            print("❌ Failed to load quantized model, falling back to standard loading...")
            # Fall back to standard loading
            args.quantization = None
            args.gguf_path = None
        else:
            print('✅ GGUF quantized Flux loaded successfully!')
    
    # Standard Flux loading (fallback or when quantization not requested)
    if not args.quantization and not args.gguf_path:
        base_model_path = f"black-forest-labs/FLUX.1-{args.model_variant}"
        print(f'Loading {base_model_path} model from HuggingFace...')
        
        # Use appropriate dtype for device
        if device == 'cpu':
            dtype = torch.float32
        elif device == 'mps':
            dtype = torch.float16
            print('Using float16 for MPS to optimize memory usage')
        else:  # CUDA
            dtype = torch.float16
        
        # Memory optimization
        flux_kwargs = {
            'torch_dtype': dtype,
            'low_cpu_mem_usage': True,
        }
        
        print(f'Loading Flux pipeline with dtype={dtype}, device={device}...')
        flux_pipe = FluxPipeline.from_pretrained(
            base_model_path,
            **flux_kwargs
        )
        print('Standard Flux pipeline loaded successfully!')
    
    # Initialize AutoStudio wrapper
    try:
        print('Initializing AUTOSTUDIOFLUX wrapper...')
        autostudio = AUTOSTUDIOFLUX(flux_pipe, image_encoder_path, ip_ckpt, device)
        print('Successfully loaded Flux model!')
    except Exception as e:
        print(f'Error initializing AUTOSTUDIOFLUX: {e}')
        print('Falling back to CPU device...')
        device = 'cpu'
        flux_pipe = flux_pipe.to('cpu')
        autostudio = AUTOSTUDIOFLUX(flux_pipe, image_encoder_path, ip_ckpt, device)
        print('Successfully loaded Flux model on CPU')

elif args.sd_version == 'fluxplus':
    base_model_path = "black-forest-labs/FLUX.1-dev"
    image_encoder_path = "/IP-Adapter/models/image_encoder/"
    ip_ckpt = "/IP-Adapter/models/ip-adapter-plus_sd15.bin"  # Will need Flux-specific adapter
    
    flux_pipe = FluxPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
    )
    autostudio = AUTOSTUDIOFLUXPlus(flux_pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
    print('Successfully loaded Flux Plus model')

with open(args.data_path, 'r', encoding='utf-8') as file:
    instructions = json.load(file)

ind = 0
for dialogue in instructions: 

    os.makedirs(f"{args.save_dir}/{dialogue}", exist_ok=True)
    start_time = time.time()

    # '0' stand for a place holder image. If you need a human-specified reference, please read the image and import it into the corresponding ID, like {,..., '1': img1}
    character_database = {'0':Image.new('RGB', (100, 100), (255, 255, 255))} 
    
    for turn in [f'turn {i+1}' for i in range(15)]:
        print(dialogue, turn)
        save_turn_dir = f"{args.save_dir}/{dialogue}/{turn}"
        try:
            instruct_now = instructions[dialogue][turn]
        except Exception as err:
            print('skip')
            continue
        
        if os.path.exists(save_turn_dir):
            continue

        os.makedirs(save_turn_dir, exist_ok=True)

        # convert prompt book
        try:
            size = instruct_now['size']
        except:
            size = [1024,1024]
        prompt = instruct_now['caption']
        bg_prompt = instruct_now['background']
        neg_prompt = instruct_now['negative']
        obj_ids = []
        gen_boxes = []  
        all_layouts = []    
        for boundingbox in instruct_now['objects']:
            if is_CMIGBENCH:
                boundingbox[1] = [element * 2 for element in boundingbox[1]]
            gen_box = [boundingbox[0],tuple(boundingbox[1])]
            obj_ids.append(str(boundingbox[2]))
            gen_boxes.append(tuple(gen_box))
            all_layouts.append(tuple(boundingbox[1]))

        prompt_book = {
            "size": size, #[height,width]
            "prompt": prompt,
            "gen_boxes": gen_boxes,
            "bg_prompt": bg_prompt,
            "global_prompt": '',
            "extra_neg_prompt": neg_prompt,
            "obj_ids":obj_ids,
            "have_ref":obj_ids.copy(), 
        }

        prompt_book = get_global_prompt(prompt_book)

        show_boxes(
            size,
            gen_boxes,
            bg_prompt=bg_prompt,
            neg_prompt=neg_prompt,
            save_dir=save_turn_dir,
        )

        if args.freeze_dialogue_seed != None: #freeze seed for each dialogue
            original_ind_base = args.freeze_dialogue_seed
        else:
            original_ind_base = ind

        for repeat_ind in range(args.repeats):
            ind_offset = repeat_ind * LARGE_CONSTANT2 + args.seed_offset
            vis_location = [dialogue, turn]

            # Adjust parameters based on model version
            if args.sd_version in ['flux', 'fluxplus']:
                # Flux-optimized parameters
                output = autostudio.generate(
                                            GROUNDING_DINO_MODEL,
                                            EFFICIENT_SAM_MODEL,
                                            character_database, 
                                            prompt_book, 
                                            do_latent_guidance,
                                            refine_step=4,
                                            num_samples=1, num_inference_steps=4, seed=ind_offset,
                                            guidance_scale=3.5,
                                            img_scale=0.7,
                                            fuse_scale=[1.2, 0],
                                            height = prompt_book['size'][0],
                                            width = prompt_book['size'][1],
                                            is_editing = False,
                                            repeat_ind = repeat_ind,
                                            )
            else:
                # SD parameters
                output = autostudio.generate(
                                            GROUNDING_DINO_MODEL,
                                            EFFICIENT_SAM_MODEL,
                                            character_database, 
                                            prompt_book, 
                                            do_latent_guidance,
                                            refine_step=20,
                                            num_samples=1, num_inference_steps=30, seed=ind_offset,
                                            img_scale=0.7,
                                            fuse_scale=[1.2, 0],
                                            height = prompt_book['size'][0],
                                            width = prompt_book['size'][1],
                                            is_editing = False,
                                            repeat_ind = repeat_ind,
                                            )
            output, character_database = output[0][0], output[1]
            show_image(output, "img", repeat_ind, 25, save_turn_dir) 

    end_time = time.time()
    use_time = end_time - start_time
    print("single dialogue time:",use_time)
