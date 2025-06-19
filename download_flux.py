#!/usr/bin/env python3
"""
Download Flux.1 models after HF authentication
"""

import os
import sys
from huggingface_hub import login, HfApi
from diffusers import FluxPipeline
import torch

def check_hf_authentication():
    """Check if user is authenticated with Hugging Face"""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False

def download_flux_model(model_name, save_path):
    """Download and save a Flux model"""
    try:
        print(f"📥 Downloading {model_name}...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir="./models/cache"
        )
        
        print(f"💾 Saving to {save_path}...")
        pipe.save_pretrained(save_path)
        
        print(f"✅ {model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    print("🚀 Flux.1 Model Downloader")
    print("=" * 50)
    
    # Check authentication
    if not check_hf_authentication():
        print("\n🔐 Authentication Required:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' permissions")
        print("3. Run one of these commands:")
        print("   export HUGGING_FACE_HUB_TOKEN='your_token_here'")
        print("   OR")
        print("   huggingface-cli login")
        print("\n4. Then run this script again")
        return False
    
    print("\n📋 Available Models:")
    models = [
        ("black-forest-labs/FLUX.1-dev", "./models/flux-dev", "High quality generation (default)"),
        ("black-forest-labs/FLUX.1-schnell", "./models/flux-schnell", "Fast 4-step generation"),
    ]
    
    for i, (model_name, path, description) in enumerate(models, 1):
        print(f"   {i}. {model_name}")
        print(f"      {description}")
        print(f"      Save to: {path}")
    
    print(f"\n🎯 Downloading models...")
    
    success_count = 0
    for model_name, save_path, description in models:
        print(f"\n" + "=" * 30)
        if download_flux_model(model_name, save_path):
            success_count += 1
    
    print(f"\n" + "=" * 50)
    print(f"📊 Download Summary: {success_count}/{len(models)} successful")
    
    if success_count > 0:
        print(f"\n🎉 Ready to use Flux.1!")
        print(f"Update your run.py model paths:")
        for model_name, save_path, description in models:
            if "schnell" in model_name:
                print(f"   For flux: base_model_path = '{save_path}'")
            else:
                print(f"   For fluxplus: base_model_path = '{save_path}'")
    
    return success_count > 0

if __name__ == "__main__":
    # Check for token in environment
    token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        try:
            login(token=token)
            print("✅ Using token from environment variable")
        except Exception as e:
            print(f"❌ Token authentication failed: {e}")
    
    main()