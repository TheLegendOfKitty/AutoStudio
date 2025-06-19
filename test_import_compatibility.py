#!/usr/bin/env python3
"""
Test script to verify import compatibility with different diffusers versions
"""

import sys
import traceback

def test_imports():
    """Test critical imports that might fail in Colab"""
    print("Testing import compatibility...")
    
    try:
        print("Testing basic imports...")
        import torch
        import numpy as np
        print("✅ Basic imports OK")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        print("Testing diffusers imports...")
        import diffusers
        print(f"✅ Diffusers version: {diffusers.__version__}")
    except Exception as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    try:
        print("Testing model imports...")
        from model.unet_2d_condition import UNet2DConditionModel
        print("✅ UNet2DConditionModel import OK")
    except Exception as e:
        print(f"❌ UNet2DConditionModel import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("Testing pipeline imports...")
        from model.pipeline_stable_diffusion import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline import OK")
    except Exception as e:
        print(f"❌ StableDiffusionPipeline import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("Testing unet_2d_blocks imports...")
        from model.unet_2d_blocks import UNetMidBlock2DCrossAttn
        print("✅ UNet2D blocks import OK")
    except Exception as e:
        print(f"❌ UNet2D blocks import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("Testing autostudio imports...")
        from model.autostudio import AUTOSTUDIO
        print("✅ AUTOSTUDIO import OK")
    except Exception as e:
        print(f"❌ AUTOSTUDIO import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        print("Testing run.py imports...")
        # Test the imports that run.py would use
        if torch.backends.mps.is_available():
            print("✅ MPS available")
        else:
            print("⚠️  MPS not available (normal for non-Apple hardware)")
        
        if torch.cuda.is_available():
            print("✅ CUDA available")
        else:
            print("⚠️  CUDA not available")
        
        print("✅ Device detection OK")
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False
    
    print("\n🎉 All import tests passed!")
    return True

def main():
    """Main test function"""
    print("AutoStudio Import Compatibility Test\n")
    
    print("=== Environment Info ===")
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except:
        print("PyTorch not available")
    
    try:
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")
    except:
        print("Diffusers not available")
    
    print("\n=== Import Tests ===")
    success = test_imports()
    
    if success:
        print("\n✅ AutoStudio should work in this environment!")
    else:
        print("\n❌ Some compatibility issues found. Check error messages above.")

if __name__ == "__main__":
    main()