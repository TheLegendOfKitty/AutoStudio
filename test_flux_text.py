#!/usr/bin/env python3
"""
Test script for Flux text-in-image generation
Demonstrates how to use Flux's special text rendering capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flux_text_generation import FluxTextGenerator, integrate_with_autostudio

def demo_text_generation():
    """Demo Flux's text generation capabilities"""
    
    print("🎨 Flux Text-in-Image Generation Demo")
    print("=" * 50)
    
    text_gen = FluxTextGenerator()
    
    print("\n📝 Available Text Styles:")
    for style, description in text_gen.text_styles.items():
        print(f"  • {style}: {description}")
    
    print("\n📍 Available Placements:")
    for placement, description in text_gen.text_placements.items():
        print(f"  • {placement}: {description}")
    
    print("\n🎯 Example Prompts:")
    examples = text_gen.create_examples()
    
    for name, prompt in examples.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  {prompt[:100]}...")
    
    print(f"\n⚙️  Optimal Flux Parameters for Text:")
    params = text_gen.optimize_for_flux_parameters()
    for key, value in params.items():
        print(f"  • {key}: {value}")
    
    print(f"\n🚀 To Generate Text Images:")
    print(f"  1. Use the examples: python run.py --sd_version flux --data_path text_examples.json")
    print(f"  2. Create custom prompts with the FluxTextGenerator class")
    print(f"  3. Use guidance_scale=3.0 for best text results")

def create_custom_example():
    """Create a custom text generation example"""
    
    text_gen = FluxTextGenerator()
    
    # Create a coffee shop sign
    coffee_prompt = text_gen.create_sign_prompt(
        scene="cozy coffee shop storefront, morning light",
        sign_text="Blue Moon Café", 
        sign_type="rustic wooden",
        text_style="carved"
    )
    
    print(f"\n☕ Custom Coffee Shop Example:")
    print(f"   {coffee_prompt}")
    
    # Create a motivational poster
    motivation_prompt = text_gen.create_poster_prompt(
        poster_style="inspirational motivational",
        main_text="NEVER GIVE UP",
        color_scheme="warm sunset colors"
    )
    
    print(f"\n💪 Motivational Poster Example:")
    print(f"   {motivation_prompt}")

if __name__ == "__main__":
    demo_text_generation()
    create_custom_example()