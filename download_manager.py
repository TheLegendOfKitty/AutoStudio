#!/usr/bin/env python3
"""
AutoStudio Download Manager

Unified, efficient model downloader that:
- Downloads only necessary components
- Avoids file splits and redundant cache
- Provides progress tracking and storage optimization
- Supports multiple model types
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from huggingface_hub import login, HfApi
from download_efficient import EfficientModelDownloader
from optimize_models import ModelOptimizer

class AutoStudioDownloadManager:
    """Unified download manager for AutoStudio models"""
    
    def __init__(self):
        self.downloader = EfficientModelDownloader()
        self.optimizer = ModelOptimizer()
        self.check_authentication()
        
    def check_authentication(self):
        """Check HuggingFace authentication"""
        try:
            api = HfApi()
            user_info = api.whoami()
            print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
            self.authenticated = True
        except Exception:
            print("⚠️  Not authenticated with HuggingFace")
            self.authenticated = False
    
    def authenticate_if_needed(self):
        """Prompt for authentication if needed"""
        if not self.authenticated:
            print("\n🔐 Some models require HuggingFace authentication")
            print("Options:")
            print("1. Set environment variable: export HF_TOKEN='your_token'")
            print("2. Use huggingface-cli login")
            print("3. Continue with public models only")
            
            choice = input("Continue anyway? (y/n): ").lower().strip()
            if choice != 'y':
                print("Please authenticate and try again.")
                sys.exit(1)
    
    def show_model_options(self):
        """Display available model options"""
        print("\n📋 Available Models:")
        print("=" * 60)
        
        models = [
            {
                "id": "flux-single",
                "name": "Flux.1-schnell (Single File)",
                "size": "~24 GB",
                "description": "Complete model in one file - faster, cleaner",
                "auth_required": True
            },
            {
                "id": "flux-minimal",
                "name": "Flux.1-schnell (Minimal Components)",
                "size": "~3-4 GB",
                "description": "Essential components only - reduced functionality",
                "auth_required": True
            },
            {
                "id": "sd15-single",
                "name": "Stable Diffusion 1.5 (Single File)",
                "size": "~4 GB", 
                "description": "Complete SD 1.5 model in one file",
                "auth_required": False
            },
            {
                "id": "sd15-minimal", 
                "name": "Stable Diffusion 1.5 (Minimal)",
                "size": "~2-3 GB",
                "description": "Essential components only",
                "auth_required": False
            },
            {
                "id": "sdxl-single",
                "name": "Stable Diffusion XL (Single File)",
                "size": "~7 GB",
                "description": "Complete SDXL model in one file",
                "auth_required": False
            },
            {
                "id": "sdxl-minimal",
                "name": "Stable Diffusion XL (Minimal)", 
                "size": "~5-6 GB",
                "description": "Essential components only",
                "auth_required": False
            }
        ]
        
        for i, model in enumerate(models, 1):
            auth_icon = "🔒" if model["auth_required"] else "🌐"
            print(f"{i}. {auth_icon} {model['name']}")
            print(f"   Size: {model['size']}")
            print(f"   {model['description']}")
            print()
        
        return models
    
    def download_model(self, model_id: str) -> Optional[str]:
        """Download a specific model"""
        try:
            # Import single file downloader
            from download_single_file import SingleFileDownloader
            single_downloader = SingleFileDownloader()
            
            if model_id == "flux-single":
                if not self.authenticated:
                    self.authenticate_if_needed()
                return single_downloader.download_flux_single_file("schnell")
                
            elif model_id == "flux-minimal":
                if not self.authenticated:
                    self.authenticate_if_needed()
                return self.downloader.download_minimal_flux()
                
            elif model_id == "sd15-single":
                return single_downloader.download_sd_single_file("sd15")
                
            elif model_id == "sd15-minimal":
                return self.downloader.download_stable_diffusion_15()
                
            elif model_id == "sdxl-single":
                return single_downloader.download_sd_single_file("sdxl")
                
            elif model_id == "sdxl-minimal":
                return self.download_sdxl_minimal()
                
            else:
                print(f"❌ Unknown model ID: {model_id}")
                return None
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def download_sdxl_minimal(self) -> str:
        """Download SDXL minimal"""
        print("📥 Downloading Stable Diffusion XL (minimal)...")
        
        from huggingface_hub import hf_hub_download
        
        model_dir = Path("./models_efficient/sdxl-minimal")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # Essential SDXL files
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json", 
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer_2/tokenizer_config.json",
            "unet/config.json",
            "vae/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors",
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
        
        return str(model_dir)
    
    def interactive_download(self):
        """Interactive download interface"""
        print("🚀 AutoStudio Download Manager")
        print("=" * 60)
        
        # Show current storage
        print("\n📊 Current Storage:")
        self.optimizer.analyze_model_storage(["./models", "./models_efficient"])
        
        # Show available models
        models = self.show_model_options()
        
        print("Additional Options:")
        print(f"{len(models)+1}. Optimize existing downloads")
        print(f"{len(models)+2}. Clean up cache")
        print(f"{len(models)+3}. List downloaded models")
        
        try:
            choice = int(input(f"\nSelect option (1-{len(models)+3}): "))
            
            if 1 <= choice <= len(models):
                model = models[choice-1]
                print(f"\n📥 Downloading {model['name']}...")
                
                model_path = self.download_model(model['id'])
                if model_path:
                    print(f"\n🎉 Download complete!")
                    print(f"📁 Model saved to: {model_path}")
                    self.show_usage_instructions(model['id'], model_path)
                    
            elif choice == len(models) + 1:
                # Optimize existing downloads
                print(f"\n🔧 Optimizing existing downloads...")
                self.optimizer.optimize_all(dry_run=False)
                
            elif choice == len(models) + 2:
                # Clean cache
                print(f"\n🧹 Cleaning cache...")
                self.downloader.cleanup_old_cache()
                
            elif choice == len(models) + 3:
                # List models
                self.downloader.list_downloaded_models()
                
            else:
                print("❌ Invalid choice")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
    
    def show_usage_instructions(self, model_id: str, model_path: str):
        """Show usage instructions for downloaded model"""
        print(f"\n📖 Usage Instructions:")
        print("=" * 40)
        
        if model_id in ["flux-single", "flux-minimal"]:
            print(f"📄 Model downloaded: {model_path}")
            if model_id == "flux-single":
                print(f"✅ Complete single-file model - full functionality")
            else:
                print(f"⚠️  Minimal components - some features may be limited")
            print(f"\n🔧 Usage:")
            print(f"   python run.py --sd_version flux --device auto")
            
        elif model_id in ["sd15-single", "sd15-minimal"]:
            print(f"📄 Model downloaded: {model_path}")
            if model_id == "sd15-single":
                print(f"✅ Complete single-file model")
            print(f"\n🔧 Usage:")
            print(f"   python run.py --sd_version 1.5 --device auto")
            
        elif model_id in ["sdxl-single", "sdxl-minimal"]:
            print(f"📄 Model downloaded: {model_path}")
            if model_id == "sdxl-single":
                print(f"✅ Complete single-file model")
            print(f"\n🔧 Usage:")
            print(f"   python run.py --sd_version xl --device auto")
        
        # Check if existing single file is already present
        existing_single_file = Path("./flux1-schnell.safetensors")
        if existing_single_file.exists() and "flux" in model_id:
            print(f"\n💡 Found existing single file: {existing_single_file}")
            print(f"   Size: {existing_single_file.stat().st_size / (1024**3):.1f} GB")
            print(f"   You can use this instead by updating run.py")
        
        print(f"\n💡 Tips:")
        print(f"   • Use --device mps on Apple Silicon for GPU acceleration") 
        print(f"   • Single-file models load faster and are easier to manage")
        print(f"   • Keep both minimal and single-file versions for different use cases")
    
    def batch_download(self, model_ids: List[str]):
        """Download multiple models in batch"""
        print(f"📦 Batch downloading {len(model_ids)} models...")
        
        successful_downloads = []
        failed_downloads = []
        
        for model_id in model_ids:
            print(f"\n{'='*30}")
            print(f"Downloading {model_id}...")
            
            model_path = self.download_model(model_id)
            if model_path:
                successful_downloads.append((model_id, model_path))
            else:
                failed_downloads.append(model_id)
        
        # Summary
        print(f"\n📊 Batch Download Summary")
        print("=" * 40)
        print(f"✅ Successful: {len(successful_downloads)}")
        print(f"❌ Failed: {len(failed_downloads)}")
        
        if successful_downloads:
            print(f"\nSuccessful downloads:")
            for model_id, path in successful_downloads:
                print(f"  • {model_id}: {path}")
        
        if failed_downloads:
            print(f"\nFailed downloads:")
            for model_id in failed_downloads:
                print(f"  • {model_id}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AutoStudio Download Manager")
    parser.add_argument("--model", "-m", help="Model to download (flux-minimal, sd15-minimal, sdxl-minimal)")
    parser.add_argument("--batch", nargs="+", help="Download multiple models")
    parser.add_argument("--optimize", action="store_true", help="Optimize existing downloads")
    parser.add_argument("--clean", action="store_true", help="Clean cache")
    parser.add_argument("--list", action="store_true", help="List downloaded models")
    
    args = parser.parse_args()
    
    # Check for HF token
    token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        try:
            login(token=token)
            print("✅ Using HF token from environment")
        except Exception as e:
            print(f"⚠️  Token authentication failed: {e}")
    
    manager = AutoStudioDownloadManager()
    
    if args.model:
        # Single model download
        model_path = manager.download_model(args.model)
        if model_path:
            manager.show_usage_instructions(args.model, model_path)
            
    elif args.batch:
        # Batch download
        manager.batch_download(args.batch)
        
    elif args.optimize:
        # Optimize existing
        manager.optimizer.optimize_all(dry_run=False)
        
    elif args.clean:
        # Clean cache
        manager.downloader.cleanup_old_cache()
        
    elif args.list:
        # List models
        manager.downloader.list_downloaded_models()
        
    else:
        # Interactive mode
        manager.interactive_download()

if __name__ == "__main__":
    main()