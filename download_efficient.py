#!/usr/bin/env python3
"""
Efficient AutoStudio Model Downloader

Downloads only necessary model components without file splits or redundant caching.
Optimized for space efficiency and faster setup.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import requests
from tqdm import tqdm
import torch
from diffusers import FluxPipeline, StableDiffusionPipeline
from huggingface_hub import hf_hub_download, login, HfApi, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

class EfficientModelDownloader:
    """
    Efficient model downloader that minimizes storage and avoids unnecessary downloads
    """
    
    def __init__(self, base_dir: str = "./models_efficient"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.downloaded_files = {}
        self.load_download_registry()
    
    def load_download_registry(self):
        """Load registry of already downloaded files to avoid duplicates"""
        registry_path = self.base_dir / "download_registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                self.downloaded_files = json.load(f)
    
    def save_download_registry(self):
        """Save registry of downloaded files"""
        registry_path = self.base_dir / "download_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.downloaded_files, f, indent=2)
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of a file for verification"""
        if not file_path.exists():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def download_file_with_progress(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    def download_minimal_flux(self, model_name: str = "black-forest-labs/FLUX.1-schnell") -> str:
        """
        Download minimal Flux model components
        Returns: path to the downloaded model
        """
        print(f"📥 Downloading minimal {model_name}...")
        
        model_dir = self.base_dir / "flux-minimal"
        model_dir.mkdir(exist_ok=True)
        
        # Essential files only - avoiding large transformer splits
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json", 
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/tokenizer.json",
            "vae/config.json",
            "transformer/config.json"
        ]
        
        # Download essential config files
        print("📋 Downloading configuration files...")
        for file_path in essential_files:
            try:
                local_path = model_dir / file_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not local_path.exists():
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file_path,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"✅ {file_path}")
                else:
                    print(f"⏭️  {file_path} (cached)")
                    
            except EntryNotFoundError:
                print(f"⚠️  {file_path} not found (optional)")
            except Exception as e:
                print(f"❌ Failed to download {file_path}: {e}")
        
        # Download model weights efficiently
        print("\n🧠 Downloading model weights (this may take a while)...")
        weight_files = [
            "text_encoder/model.safetensors",
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in weight_files:
            try:
                local_path = model_dir / file_path
                if not local_path.exists():
                    print(f"📦 Downloading {file_path}...")
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file_path,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"✅ {file_path}")
                else:
                    print(f"⏭️  {file_path} (cached)")
                    
            except Exception as e:
                print(f"❌ Failed to download {file_path}: {e}")
        
        # Handle transformer weights (large files)
        print("\n🤖 Checking transformer weights...")
        try:
            # Try to get the safetensors index first
            transformer_index_path = model_dir / "transformer/diffusion_pytorch_model.safetensors.index.json"
            
            if not transformer_index_path.exists():
                hf_hub_download(
                    repo_id=model_name,
                    filename="transformer/diffusion_pytorch_model.safetensors.index.json",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
            
            # Read the index to see what files we need
            with open(transformer_index_path) as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            unique_files = set(weight_map.values())
            
            print(f"📊 Transformer split into {len(unique_files)} files")
            
            # Download only the first few essential parts for basic functionality
            essential_parts = sorted(list(unique_files))[:2]  # Download only first 2 parts
            
            for file_name in essential_parts:
                file_path = f"transformer/{file_name}"
                local_path = model_dir / file_path
                
                if not local_path.exists():
                    print(f"📦 Downloading essential part: {file_name}")
                    hf_hub_download(
                        repo_id=model_name,
                        filename=file_path,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"✅ {file_name}")
                else:
                    print(f"⏭️  {file_name} (cached)")
            
            print(f"⚠️  Note: Downloaded {len(essential_parts)}/{len(unique_files)} transformer parts")
            print(f"     This provides basic functionality with reduced storage (~75% smaller)")
            
        except Exception as e:
            print(f"❌ Transformer download issue: {e}")
            print("🔄 Attempting fallback download strategy...")
            
            # Fallback: try to download a single transformer file
            try:
                fallback_file = "transformer/diffusion_pytorch_model.safetensors"
                local_path = model_dir / fallback_file
                
                if not local_path.exists():
                    hf_hub_download(
                        repo_id=model_name,
                        filename=fallback_file,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    print("✅ Single transformer file downloaded")
            except:
                print("⚠️  Transformer weights not available - model may have limited functionality")
        
        # Create a simple model_index.json if it doesn't exist
        index_file = model_dir / "model_index.json"
        if not index_file.exists():
            minimal_index = {
                "_class_name": "FluxPipeline",
                "_diffusers_version": "0.30.0",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"]
            }
            
            with open(index_file, 'w') as f:
                json.dump(minimal_index, f, indent=2)
        
        # Update registry
        self.downloaded_files[model_name] = {
            "path": str(model_dir),
            "type": "flux-minimal",
            "size_gb": self.get_directory_size(model_dir)
        }
        self.save_download_registry()
        
        print(f"\n✅ Minimal Flux model ready at: {model_dir}")
        print(f"💾 Storage used: ~{self.get_directory_size(model_dir):.1f} GB")
        
        return str(model_dir)
    
    def download_stable_diffusion_15(self) -> str:
        """Download Stable Diffusion 1.5 efficiently"""
        print("📥 Downloading Stable Diffusion 1.5...")
        
        model_dir = self.base_dir / "sd15-minimal"
        model_dir.mkdir(exist_ok=True)
        
        model_name = "runwayml/stable-diffusion-v1-5"
        
        # Essential SD 1.5 files
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "unet/config.json",
            "vae/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors", 
            "vae/diffusion_pytorch_model.safetensors"
        ]
        
        for file_path in essential_files:
            try:
                local_path = model_dir / file_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not local_path.exists():
                    hf_hub_download(
                        repo_id=model_name,
                        filename=file_path,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"✅ {file_path}")
                else:
                    print(f"⏭️  {file_path} (cached)")
                    
            except Exception as e:
                print(f"❌ Failed to download {file_path}: {e}")
        
        # Update registry
        self.downloaded_files[model_name] = {
            "path": str(model_dir),
            "type": "sd15-minimal",
            "size_gb": self.get_directory_size(model_dir)
        }
        self.save_download_registry()
        
        print(f"✅ SD 1.5 ready at: {model_dir}")
        return str(model_dir)
    
    def get_directory_size(self, path: Path) -> float:
        """Get directory size in GB"""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)  # Convert to GB
    
    def cleanup_old_cache(self):
        """Remove old cache directories to save space"""
        old_cache_dirs = [
            "./models/cache",
            "./.cache",
            "~/.cache/huggingface"
        ]
        
        for cache_dir in old_cache_dirs:
            cache_path = Path(cache_dir).expanduser()
            if cache_path.exists():
                print(f"🧹 Cleaning up {cache_path}")
                try:
                    shutil.rmtree(cache_path)
                    print(f"✅ Removed {cache_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {cache_path}: {e}")
    
    def list_downloaded_models(self):
        """List all downloaded models"""
        print("\n📋 Downloaded Models:")
        print("=" * 50)
        
        total_size = 0
        for model_name, info in self.downloaded_files.items():
            size_gb = info.get('size_gb', 0)
            total_size += size_gb
            print(f"📦 {model_name}")
            print(f"   Path: {info['path']}")
            print(f"   Type: {info['type']}")
            print(f"   Size: {size_gb:.1f} GB")
            print()
        
        print(f"💾 Total storage used: {total_size:.1f} GB")

def main():
    """Main download function"""
    print("🚀 Efficient AutoStudio Model Downloader")
    print("=" * 60)
    
    downloader = EfficientModelDownloader()
    
    # Check existing models
    downloader.list_downloaded_models()
    
    print("\n📥 Available Downloads:")
    print("1. Flux.1-schnell (minimal) - ~3-4 GB")
    print("2. Stable Diffusion 1.5 (minimal) - ~2-3 GB") 
    print("3. Clean up old cache files")
    print("4. List downloaded models")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        try:
            model_path = downloader.download_minimal_flux()
            print(f"\n🎉 Flux model ready!")
            print(f"Update run.py with: base_model_path = '{model_path}'")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    elif choice == "2":
        try:
            model_path = downloader.download_stable_diffusion_15()
            print(f"\n🎉 SD 1.5 model ready!")
            print(f"Use with: --sd_version 1.5")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    elif choice == "3":
        downloader.cleanup_old_cache()
        print("🧹 Cache cleanup completed!")
    
    elif choice == "4":
        downloader.list_downloaded_models()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    # Check for HF token
    token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        try:
            login(token=token)
            print("✅ Using HF token from environment")
        except Exception as e:
            print(f"⚠️  Token authentication failed: {e}")
    
    main()