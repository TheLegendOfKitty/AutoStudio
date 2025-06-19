#!/usr/bin/env python3
"""
Single-File Model Downloader for AutoStudio

Downloads complete models as single .safetensors files instead of split components.
Much more efficient than downloading multiple split files.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import hashlib
from huggingface_hub import hf_hub_download, login, HfApi
from huggingface_hub.utils import EntryNotFoundError

class SingleFileDownloader:
    """Download complete models as single safetensors files"""
    
    def __init__(self, base_dir: str = "./models_single"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.registry_file = self.base_dir / "downloaded_models.json"
        self.load_registry()
    
    def load_registry(self):
        """Load registry of downloaded models"""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
    
    def save_registry(self):
        """Save registry of downloaded models"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_file_size_gb(self, file_path: Path) -> float:
        """Get file size in GB"""
        if file_path.exists():
            return file_path.stat().st_size / (1024**3)
        return 0
    
    def verify_file_integrity(self, file_path: Path, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file integrity"""
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        
        # Check if file size is reasonable (not corrupted/incomplete)
        if expected_size and abs(file_size - expected_size) > 1024*1024:  # Allow 1MB difference
            return False
        
        # Check if file is not empty
        if file_size < 1024:  # Less than 1KB is suspicious
            return False
        
        return True
    
    def download_flux_single_file(self, model_variant: str = "dev") -> Optional[str]:
        """
        Download Flux model as single safetensors file
        
        Args:
            model_variant: "dev" (high quality, default) or "schnell" (fast)
        """
        if model_variant == "schnell":
            repo_id = "black-forest-labs/FLUX.1-schnell"
            filename = "flux1-schnell.safetensors"
            expected_size_gb = 23.8  # Approximate size
        elif model_variant == "dev":
            repo_id = "black-forest-labs/FLUX.1-dev" 
            filename = "flux1-dev.safetensors"
            expected_size_gb = 23.8
        else:
            print(f"❌ Unknown Flux variant: {model_variant}")
            return None
        
        print(f"📥 Downloading Flux.1-{model_variant} as single file...")
        print(f"   Model: {repo_id}")
        print(f"   File: {filename}")
        print(f"   Expected size: ~{expected_size_gb:.1f} GB")
        
        local_file = self.base_dir / filename
        
        # Check if already downloaded
        if local_file.exists() and self.verify_file_integrity(local_file):
            current_size = self.get_file_size_gb(local_file)
            print(f"✅ Already downloaded: {local_file}")
            print(f"   Current size: {current_size:.1f} GB")
            
            # Update registry
            self.registry[f"flux-{model_variant}"] = {
                "repo_id": repo_id,
                "filename": filename,
                "local_path": str(local_file),
                "size_gb": current_size,
                "type": "single-file"
            }
            self.save_registry()
            return str(local_file)
        
        try:
            print(f"🔍 Checking model availability...")
            
            # Download the single file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.base_dir,
                local_dir_use_symlinks=False,
                resume_download=True  # Resume if partially downloaded
            )
            
            # Verify download
            if self.verify_file_integrity(Path(downloaded_path)):
                file_size = self.get_file_size_gb(Path(downloaded_path))
                print(f"✅ Download complete: {downloaded_path}")
                print(f"   Final size: {file_size:.1f} GB")
                
                # Update registry
                self.registry[f"flux-{model_variant}"] = {
                    "repo_id": repo_id,
                    "filename": filename,
                    "local_path": downloaded_path,
                    "size_gb": file_size,
                    "type": "single-file"
                }
                self.save_registry()
                
                return downloaded_path
            else:
                print(f"❌ Download verification failed")
                return None
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            
            # Check if it's an authentication issue
            if "401" in str(e) or "authentication" in str(e).lower():
                print(f"\n🔐 Authentication required for {repo_id}")
                print(f"Please visit: https://huggingface.co/{repo_id}")
                print(f"1. Request access if needed")
                print(f"2. Get your HF token: https://huggingface.co/settings/tokens")
                print(f"3. Set: export HF_TOKEN='your_token_here'")
            
            return None
    
    def download_sd_single_file(self, model_type: str = "sd15") -> Optional[str]:
        """
        Download Stable Diffusion as single file
        
        Args:
            model_type: "sd15" or "sdxl"
        """
        if model_type == "sd15":
            # Try popular single-file SD 1.5 models
            variants = [
                ("runwayml/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
                ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned.safetensors"),
            ]
            expected_size_gb = 4.2
        elif model_type == "sdxl":
            variants = [
                ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
            ]
            expected_size_gb = 6.9
        else:
            print(f"❌ Unknown SD model type: {model_type}")
            return None
        
        print(f"📥 Downloading {model_type.upper()} as single file...")
        
        for repo_id, filename in variants:
            print(f"\n🔍 Trying {repo_id}/{filename}...")
            
            local_file = self.base_dir / f"{model_type}_{filename}"
            
            # Check if already downloaded
            if local_file.exists() and self.verify_file_integrity(local_file):
                current_size = self.get_file_size_gb(local_file)
                print(f"✅ Already downloaded: {local_file}")
                print(f"   Current size: {current_size:.1f} GB")
                return str(local_file)
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.base_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                
                # Move to standardized name
                Path(downloaded_path).rename(local_file)
                
                if self.verify_file_integrity(local_file):
                    file_size = self.get_file_size_gb(local_file)
                    print(f"✅ Download complete: {local_file}")
                    print(f"   Final size: {file_size:.1f} GB")
                    
                    # Update registry
                    self.registry[model_type] = {
                        "repo_id": repo_id,
                        "filename": filename,
                        "local_path": str(local_file),
                        "size_gb": file_size,
                        "type": "single-file"
                    }
                    self.save_registry()
                    
                    return str(local_file)
                
            except Exception as e:
                print(f"❌ Failed to download from {repo_id}: {e}")
                continue
        
        print(f"❌ Could not download {model_type} from any source")
        return None
    
    def list_available_downloads(self):
        """List available single-file downloads"""
        print("\n📋 Available Single-File Downloads:")
        print("=" * 50)
        
        downloads = [
            {
                "id": "flux-schnell",
                "name": "Flux.1-schnell",
                "size": "~24 GB",
                "description": "Fast 4-step generation (single file)",
                "auth_required": True
            },
            {
                "id": "flux-dev", 
                "name": "Flux.1-dev",
                "size": "~24 GB",
                "description": "High quality generation (single file)",
                "auth_required": True
            },
            {
                "id": "sd15",
                "name": "Stable Diffusion 1.5",
                "size": "~4 GB", 
                "description": "Classic SD model (single file)",
                "auth_required": False
            },
            {
                "id": "sdxl",
                "name": "Stable Diffusion XL",
                "size": "~7 GB",
                "description": "High resolution model (single file)", 
                "auth_required": False
            }
        ]
        
        for i, download in enumerate(downloads, 1):
            auth_icon = "🔒" if download["auth_required"] else "🌐"
            print(f"{i}. {auth_icon} {download['name']}")
            print(f"   Size: {download['size']}")
            print(f"   {download['description']}")
            print()
        
        return downloads
    
    def list_downloaded_models(self):
        """List downloaded models"""
        print("\n📦 Downloaded Models:")
        print("=" * 40)
        
        if not self.registry:
            print("No models downloaded yet.")
            return
        
        total_size = 0
        for model_id, info in self.registry.items():
            size_gb = info.get('size_gb', 0)
            total_size += size_gb
            
            print(f"📄 {model_id}")
            print(f"   File: {Path(info['local_path']).name}")
            print(f"   Path: {info['local_path']}")
            print(f"   Size: {size_gb:.1f} GB")
            print()
        
        print(f"💾 Total storage: {total_size:.1f} GB")
    
    def show_usage_instructions(self, model_id: str, file_path: str):
        """Show usage instructions"""
        print(f"\n📖 Usage Instructions for {model_id}:")
        print("=" * 40)
        
        if "flux" in model_id:
            print(f"✅ Single Flux file downloaded!")
            print(f"📁 Location: {file_path}")
            print(f"\n🔧 To use with AutoStudio:")
            print(f"1. Update run.py to point to single file:")
            print(f"   base_model_path = '{file_path}'")
            print(f"2. Run: python run.py --sd_version flux --device auto")
            
        elif model_id == "sd15":
            print(f"✅ Single SD 1.5 file downloaded!")
            print(f"📁 Location: {file_path}")
            print(f"\n🔧 To use:")
            print(f"Run: python run.py --sd_version 1.5 --device auto")
            
        elif model_id == "sdxl":
            print(f"✅ Single SDXL file downloaded!")
            print(f"📁 Location: {file_path}")
            print(f"\n🔧 To use:")
            print(f"Run: python run.py --sd_version xl --device auto")
        
        print(f"\n💡 Benefits of single-file download:")
        print(f"   • No split files to manage")
        print(f"   • Faster loading")
        print(f"   • Easy to move/backup")
        print(f"   • Less storage overhead")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single-File Model Downloader")
    parser.add_argument("--model", "-m", choices=["flux-schnell", "flux-dev", "sd15", "sdxl"], 
                       help="Model to download")
    parser.add_argument("--list", "-l", action="store_true", help="List downloaded models")
    
    args = parser.parse_args()
    
    # Check for HF token
    token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        try:
            login(token=token)
            print("✅ Using HF token from environment")
        except Exception as e:
            print(f"⚠️  Token authentication failed: {e}")
    
    downloader = SingleFileDownloader()
    
    if args.list:
        downloader.list_downloaded_models()
        return
    
    if args.model:
        # Download specific model
        if args.model.startswith("flux-"):
            variant = args.model.split("-")[1]
            file_path = downloader.download_flux_single_file(variant)
        elif args.model in ["sd15", "sdxl"]:
            file_path = downloader.download_sd_single_file(args.model)
        
        if file_path:
            downloader.show_usage_instructions(args.model, file_path)
    else:
        # Interactive mode
        print("🚀 Single-File Model Downloader")
        print("=" * 40)
        
        downloader.list_downloaded_models()
        downloads = downloader.list_available_downloads()
        
        try:
            choice = int(input(f"Select model to download (1-{len(downloads)}): "))
            if 1 <= choice <= len(downloads):
                download = downloads[choice-1]
                model_id = download["id"]
                
                print(f"\n📥 Downloading {download['name']}...")
                
                if model_id.startswith("flux-"):
                    variant = model_id.split("-")[1]
                    file_path = downloader.download_flux_single_file(variant)
                elif model_id in ["sd15", "sdxl"]:
                    file_path = downloader.download_sd_single_file(model_id)
                
                if file_path:
                    downloader.show_usage_instructions(model_id, file_path)
            else:
                print("❌ Invalid choice")
                
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()