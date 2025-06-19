from diffusers import FluxPipeline
import torch
import os
from transformers import BitsAndBytesConfig

# Create directory
os.makedirs('./models/flux-dev-4bit', exist_ok=True)

print("Downloading Flux.1-dev with 4-bit quantization...")

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_use_double_quant=True,  # Double quantization for better compression
)

pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Saving 4-bit quantized model locally...")
pipe.save_pretrained('./models/flux-dev-4bit')
print('Download complete! Saved to ./models/flux-dev-4bit')
print('Model size reduced by ~75% with 4-bit quantization!')