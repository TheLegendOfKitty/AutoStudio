#!/usr/bin/env python3
"""
Test script for enhanced Flux text encoding
Run this to test the new weighted prompt features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_flux_text import patch_autostudio_flux, create_example_prompts
import torch

def test_enhanced_flux():
    """Test the enhanced Flux text encoding with a simple example"""
    
    print("🚀 Testing Enhanced Flux Text Encoding")
    print("=" * 50)
    
    # This would normally be called after loading your AUTOSTUDIOFLUX instance
    print("\n📋 Example Enhanced Prompts:")
    examples = create_example_prompts()
    
    for i, prompt in enumerate(examples, 1):
        print(f"\n{i}. Original prompt:")
        print(f"   {prompt}")
        
        # Simulate the parsing (without actual model)
        from enhanced_flux_text import EnhancedFluxTextEncoder
        dummy_encoder = EnhancedFluxTextEncoder(None, 'cpu')
        parsed = dummy_encoder.parse_weighted_prompt(prompt)
        print(f"   Parsed result:")
        print(f"   {parsed}")
    
    print(f"\n✅ Enhanced text encoding ready!")
    print(f"\n🔧 To use with your AutoStudio:")
    print(f"   1. Load your Flux model normally:")
    print(f"      python run.py --sd_version flux --device auto --data_path enhanced_simple.json")
    print(f"   ")
    print(f"   2. Or integrate directly in Python:")
    print(f"      from enhanced_flux_text import patch_autostudio_flux")
    print(f"      # After loading autostudio...")
    print(f"      patch_autostudio_flux(autostudio)")

if __name__ == "__main__":
    test_enhanced_flux()