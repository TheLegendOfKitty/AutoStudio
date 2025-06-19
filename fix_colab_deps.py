#!/usr/bin/env python3
"""
Fix Colab dependency conflicts for AutoStudio
Resolves torchvision/transformers/diffusers version conflicts
"""

import subprocess
import sys

def run_command(cmd):
    """Run shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Running: {cmd}")
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run {cmd}: {e}")
        return False

def fix_colab_dependencies():
    """Fix dependency conflicts in Colab environment"""
    
    print("🔧 Fixing Colab Dependencies for AutoStudio")
    print("=" * 50)
    
    # Step 1: Uninstall conflicting packages
    print("\n📦 Uninstalling conflicting packages...")
    packages_to_remove = [
        "torchvision", 
        "transformers", 
        "diffusers",
        "accelerate"
    ]
    
    for package in packages_to_remove:
        print(f"Removing {package}...")
        run_command(f"pip uninstall -y {package}")
    
    # Step 2: Install compatible versions
    print("\n📥 Installing compatible versions...")
    
    # Install torch first (if needed)
    print("Installing PyTorch...")
    run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # Install compatible transformers version
    print("Installing transformers...")
    run_command("pip install transformers==4.44.0")
    
    # Install compatible diffusers version
    print("Installing diffusers...")
    run_command("pip install diffusers==0.30.0")
    
    # Install other required packages
    print("Installing additional packages...")
    packages = [
        "accelerate==0.33.0",
        "safetensors",
        "pillow",
        "numpy",
        "matplotlib",
        "opencv-python",
        "huggingface_hub",
        "psutil",
        "inflect",
        "gradio"
    ]
    
    for package in packages:
        run_command(f"pip install {package}")
    
    # Step 3: Install GGUF support
    print("\n🎯 Installing GGUF support...")
    run_command("pip install gguf")
    
    # Step 4: Verify installation
    print("\n✅ Verifying installation...")
    
    verification_script = '''
import torch
print(f"PyTorch: {torch.__version__}")

import torchvision
print(f"Torchvision: {torchvision.__version__}")

import transformers
print(f"Transformers: {transformers.__version__}")

import diffusers
print(f"Diffusers: {diffusers.__version__}")

try:
    from diffusers import FluxPipeline, AutoencoderKL
    print("✅ Diffusers imports working")
except Exception as e:
    print(f"❌ Diffusers import error: {e}")

try:
    from diffusers import GGUFQuantizationConfig
    print("✅ GGUF support available")
except Exception as e:
    print(f"❌ GGUF not available: {e}")

try:
    from transformers import AutoImageProcessor
    print("✅ AutoImageProcessor working")
except Exception as e:
    print(f"❌ AutoImageProcessor error: {e}")
'''
    
    print("Running verification...")
    with open("/tmp/verify.py", "w") as f:
        f.write(verification_script)
    
    run_command("python /tmp/verify.py")
    
    print("\n🎉 Dependency fix complete!")
    print("\n📋 Next steps:")
    print("1. Restart your runtime if in Colab: Runtime → Restart Runtime")
    print("2. Re-run your AutoStudio command")
    print("3. If issues persist, try the alternative fix below")

def alternative_fix():
    """Alternative fix using different approach"""
    
    print("\n🔄 Alternative Fix Method")
    print("=" * 30)
    
    print("If the main fix doesn't work, try this manual approach:")
    print("""
# In a new Colab cell, run these commands one by one:

!pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install --force-reinstall transformers==4.44.0
!pip install --force-reinstall diffusers==0.30.0 
!pip install --force-reinstall accelerate==0.33.0
!pip install gguf safetensors pillow numpy matplotlib opencv-python huggingface_hub psutil inflect

# Then restart runtime and test:
import torch, diffusers, transformers
print("All imports working!")
""")

if __name__ == "__main__":
    fix_colab_dependencies()
    alternative_fix()