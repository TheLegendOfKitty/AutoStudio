#!/usr/bin/env python3
"""
Flux GGUF Quantization Support
Provides GGUF quantization for memory-efficient Flux model loading
"""

import torch
import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
from diffusers import FluxPipeline, FluxTransformer2DModel
from huggingface_hub import hf_hub_download, list_repo_files

# Suppress common model loading warnings
warnings.filterwarnings("ignore", message=".*beta.*will be renamed internally to.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*will be renamed internally to.*weight.*")

try:
    from diffusers import GGUFQuantizationConfig
    GGUF_AVAILABLE = True
except ImportError:
    print("⚠️  GGUF support not available. Install with: pip install diffusers[gguf]")
    GGUF_AVAILABLE = False


class FluxGGUFManager:
    """Manages GGUF quantized Flux models"""
    
    QUANTIZATION_LEVELS = {
        "Q8_0": {
            "memory_reduction": "50%",
            "quality": "99% identical to FP16",
            "description": "Best quality, recommended for high-end systems",
            "file_suffix": "Q8_0.gguf"
        },
        "Q6_K": {
            "memory_reduction": "60%", 
            "quality": "98% identical to FP16",
            "description": "Excellent quality, good balance",
            "file_suffix": "Q6_K.gguf"
        },
        "Q5_K_S": {
            "memory_reduction": "65%",
            "quality": "96% quality, slight loss",
            "description": "Recommended for most users", 
            "file_suffix": "Q5_K_S.gguf"
        },
        "Q5_1": {
            "memory_reduction": "67%",
            "quality": "95% quality, good for most cases",
            "description": "Alternative 5-bit quantization",
            "file_suffix": "Q5_1.gguf"
        },
        "Q5_0": {
            "memory_reduction": "67%",
            "quality": "95% quality, legacy format",
            "description": "Legacy 5-bit quantization",
            "file_suffix": "Q5_0.gguf"
        },
        "Q4_K_S": {
            "memory_reduction": "70%",
            "quality": "93% quality, noticeable but acceptable",
            "description": "Good for lower-end systems",
            "file_suffix": "Q4_K_S.gguf"
        },
        "Q4_1": {
            "memory_reduction": "72%",
            "quality": "92% quality, some loss",
            "description": "Alternative 4-bit quantization",
            "file_suffix": "Q4_1.gguf"
        },
        "Q4_0": {
            "memory_reduction": "72%",
            "quality": "92% quality, legacy format",
            "description": "Legacy 4-bit quantization",
            "file_suffix": "Q4_0.gguf"
        },
        "Q3_K_S": {
            "memory_reduction": "75%",
            "quality": "88% quality, noticeable degradation",
            "description": "For very limited VRAM",
            "file_suffix": "Q3_K_S.gguf"
        },
        "Q2_K": {
            "memory_reduction": "80%",
            "quality": "75% quality, significant loss",
            "description": "Only for extremely limited systems",
            "file_suffix": "Q2_K.gguf"
        }
    }
    
    GGUF_REPOS = {
        "schnell": "city96/FLUX.1-schnell-gguf",
        "dev": "city96/FLUX.1-dev-gguf"
    }
    
    def __init__(self):
        self.cache_dir = Path.home() / ".cache" / "autostudio" / "gguf"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def show_available_quantizations(self):
        """Display available quantization levels"""
        print("\n🎯 Available GGUF Quantization Levels:")
        print("=" * 60)
        
        for level, info in self.QUANTIZATION_LEVELS.items():
            print(f"📊 {level}:")
            print(f"   Memory: {info['memory_reduction']} reduction")
            print(f"   Quality: {info['quality']}")
            print(f"   Use case: {info['description']}")
            print()
    
    def recommend_quantization(self, device: str, available_memory_gb: Optional[float] = None) -> str:
        """Recommend quantization level based on device and memory"""
        
        if device == 'cpu':
            return "Q5_K_S"  # Good balance for CPU inference
        elif device == 'mps':
            return "Q6_K"    # Mac has unified memory, can handle higher quality
        elif device == 'cuda':
            if available_memory_gb:
                if available_memory_gb >= 16:
                    return "Q8_0"    # High-end GPU
                elif available_memory_gb >= 12:
                    return "Q6_K"    # Mid-high GPU
                elif available_memory_gb >= 8:
                    return "Q5_K_S"  # Mid-range GPU
                else:
                    return "Q4_K_S"  # Low-end GPU
            else:
                return "Q5_K_S"  # Safe default
        else:
            return "Q5_K_S"  # Safe default
    
    def list_available_files(self, model_variant: str = "schnell") -> Dict[str, str]:
        """List available GGUF files for a model variant"""
        
        if not GGUF_AVAILABLE:
            print("❌ GGUF support not available")
            return {}
        
        repo_id = self.GGUF_REPOS.get(model_variant)
        if not repo_id:
            print(f"❌ Unknown model variant: {model_variant}")
            return {}
        
        try:
            files = list_repo_files(repo_id)
            gguf_files = {f: f for f in files if f.endswith('.gguf')}
            return gguf_files
            
        except Exception as e:
            print(f"❌ Error listing files from {repo_id}: {e}")
            return {}
    
    def download_gguf_model(self, 
                           model_variant: str = "schnell",
                           quantization: str = "Q5_K_M") -> Optional[str]:
        """Download GGUF model file"""
        
        if not GGUF_AVAILABLE:
            print("❌ GGUF support not available. Install with: pip install diffusers[gguf]")
            return None
        
        repo_id = self.GGUF_REPOS.get(model_variant)
        if not repo_id:
            print(f"❌ Unknown model variant: {model_variant}")
            return None
        
        # Construct filename
        if quantization in self.QUANTIZATION_LEVELS:
            suffix = self.QUANTIZATION_LEVELS[quantization]["file_suffix"]
            filename = f"flux1-{model_variant}-{suffix}"
        else:
            filename = f"flux1-{model_variant}-{quantization}.gguf"
        
        print(f"📥 Downloading {filename} from {repo_id}...")
        print(f"🎯 Quantization: {quantization}")
        
        if quantization in self.QUANTIZATION_LEVELS:
            info = self.QUANTIZATION_LEVELS[quantization]
            print(f"💾 Memory reduction: {info['memory_reduction']}")
            print(f"🎨 Quality: {info['quality']}")
        
        try:
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Downloaded: {local_file}")
            return local_file
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"💡 Available files:")
            available = self.list_available_files(model_variant)
            for file in available:
                print(f"   • {file}")
            return None
    
    def load_quantized_transformer(self, 
                                  gguf_path: str,
                                  compute_dtype: torch.dtype = torch.bfloat16) -> Optional[FluxTransformer2DModel]:
        """Load quantized transformer from GGUF file"""
        
        if not GGUF_AVAILABLE:
            print("❌ GGUF support not available")
            return None
        
        if not os.path.exists(gguf_path):
            print(f"❌ GGUF file not found: {gguf_path}")
            return None
        
        print(f"📄 Loading quantized transformer from: {gguf_path}")
        print(f"🔧 Compute dtype: {compute_dtype}")
        
        try:
            # Create quantization config
            quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
            
            # Load transformer
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=compute_dtype,
            )
            
            print(f"✅ Quantized transformer loaded successfully")
            return transformer
            
        except Exception as e:
            print(f"❌ Failed to load quantized transformer: {e}")
            return None
    
    def create_quantized_pipeline(self,
                                 model_variant: str = "schnell", 
                                 quantization: str = "Q5_K_M",
                                 device: str = "auto",
                                 gguf_path: Optional[str] = None) -> Optional[FluxPipeline]:
        """Create Flux pipeline with quantized transformer"""
        
        if not GGUF_AVAILABLE:
            print("❌ GGUF support not available")
            return None
        
        # Determine compute dtype based on device
        if device == 'cpu':
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.bfloat16
        
        # Download GGUF if path not provided
        if not gguf_path:
            gguf_path = self.download_gguf_model(model_variant, quantization)
            if not gguf_path:
                return None
        
        # Load quantized transformer
        transformer = self.load_quantized_transformer(gguf_path, compute_dtype)
        if not transformer:
            return None
        
        # Base model repo
        base_repo = f"black-forest-labs/FLUX.1-{model_variant}"
        
        print(f"🔄 Creating pipeline with base model: {base_repo}")
        
        try:
            # Create pipeline with quantized transformer
            pipe = FluxPipeline.from_pretrained(
                base_repo,
                transformer=transformer,
                torch_dtype=compute_dtype,
                low_cpu_mem_usage=True,
            )
            
            print(f"✅ Quantized pipeline created successfully")
            print(f"🎯 Model: FLUX.1-{model_variant}")
            print(f"📊 Quantization: {quantization}")
            print(f"💾 Compute dtype: {compute_dtype}")
            
            return pipe
            
        except Exception as e:
            print(f"❌ Failed to create pipeline: {e}")
            return None


def get_memory_info(device: str) -> Optional[float]:
    """Get available memory in GB for the device"""
    
    try:
        if device == 'cuda' and torch.cuda.is_available():
            memory_bytes = torch.cuda.get_device_properties(0).total_memory
            return memory_bytes / (1024**3)  # Convert to GB
        elif device == 'mps' and torch.backends.mps.is_available():
            # MPS uses unified memory, estimate based on system
            import psutil
            return psutil.virtual_memory().total / (1024**3) * 0.6  # Use 60% of system RAM
        else:
            # CPU - use system RAM
            import psutil
            return psutil.virtual_memory().available / (1024**3)
    except:
        return None


def main():
    """Demo the GGUF quantization system"""
    
    print("🎯 Flux GGUF Quantization Manager")
    print("=" * 40)
    
    manager = FluxGGUFManager()
    
    # Show available quantizations
    manager.show_available_quantizations()
    
    # Test recommendations
    devices = ['cuda', 'mps', 'cpu']
    
    print("💡 Recommended quantizations by device:")
    for device in devices:
        memory = get_memory_info(device)
        rec = manager.recommend_quantization(device, memory)
        memory_str = f"{memory:.1f}GB" if memory else "Unknown"
        print(f"   {device.upper()}: {rec} (Memory: {memory_str})")
    
    print(f"\n🔧 Usage:")
    print(f"   python run.py --sd_version flux --quantization Q5_K_M")
    print(f"   python run.py --sd_version flux --gguf_path /path/to/model.gguf")


if __name__ == "__main__":
    main()