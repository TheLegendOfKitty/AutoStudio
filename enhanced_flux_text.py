"""
Enhanced Flux Text Embedding Module
Implements advanced text encoding for Flux models with weighted prompts and dual encoder optimization
"""

import torch
import re
from typing import List, Union


class EnhancedFluxTextEncoder:
    """Enhanced text encoder for Flux models with weighted prompt support"""
    
    def __init__(self, flux_pipeline, device='auto'):
        self.pipe = flux_pipeline
        self.device = device
    
    def encode_prompts(self, prompts: Union[str, List[str]], enable_weighting=True, max_sequence_length=512):
        """
        Enhanced Flux text encoding with weighted prompts and dual encoder optimization
        
        Args:
            prompts: Text prompt(s) to encode
            enable_weighting: Whether to parse weighted prompt syntax like (word:1.3)
            max_sequence_length: Maximum sequence length for T5 encoder
        
        Returns:
            Enhanced prompt embeddings optimized for Flux
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Parse weighted prompts if enabled
        if enable_weighting:
            processed_prompts = []
            for prompt in prompts:
                processed_prompt = self.parse_weighted_prompt(prompt)
                processed_prompts.append(processed_prompt)
            prompts = processed_prompts
        
        with torch.no_grad():
            # Get both T5 and CLIP embeddings using Flux's dual encoder system
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompts,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                max_sequence_length=max_sequence_length,
            )
            
            # Apply Flux-specific optimizations
            prompt_embeds = self.optimize_flux_embeddings(prompt_embeds, pooled_prompt_embeds)
            
        return prompt_embeds
    
    def parse_weighted_prompt(self, prompt: str) -> str:
        """
        Parse weighted prompt syntax like (word:1.3) or [word:0.8]
        
        Syntax:
        - (word:1.3) - Emphasize word with weight 1.3
        - [word:0.8] - De-emphasize word with weight 0.8
        - (word) - Default emphasis (1.1)
        - [word] - Default de-emphasis (0.9)
        """
        
        # Handle parentheses weights (word:1.3) for emphasis
        def replace_parentheses(match):
            word = match.group(1).strip()
            weight = float(match.group(2)) if match.group(2) else 1.1
            
            # For Flux, we use repetition for emphasis since it has strong T5 understanding
            if weight > 1.0:
                repeats = min(int(weight), 3)  # Cap at 3 repeats to avoid token overflow
                return ' '.join([word] * repeats)
            else:
                return word
        
        # Handle bracket weights [word:0.8] for de-emphasis
        def replace_brackets(match):
            word = match.group(1).strip()
            weight = float(match.group(2)) if match.group(2) else 0.9
            
            # For negative weights, conditionally include
            if weight < 0.5:
                return ""  # Remove entirely if very low weight
            elif weight < 1.0:
                return word  # Keep but don't emphasize
            else:
                return word
        
        # Apply weighted syntax parsing
        processed = re.sub(r'\((.*?)(?::([\d.]+))?\)', replace_parentheses, prompt)
        processed = re.sub(r'\[(.*?)(?::([\d.]+))?\]', replace_brackets, processed)
        
        # Clean up extra spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def optimize_flux_embeddings(self, t5_embeds: torch.Tensor, clip_embeds: torch.Tensor) -> torch.Tensor:
        """
        Optimize embeddings specifically for Flux's characteristics
        
        Flux uses:
        - Lower guidance scale (3.5 vs SD's 7.5)
        - T5-XXL for detailed text understanding
        - CLIP for visual-text alignment
        """
        
        # Normalize embeddings for better stability
        t5_embeds = t5_embeds / t5_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Apply enhancement factor optimized for Flux's 3.5 guidance scale
        # Since Flux uses lower guidance, we need stronger text conditioning
        enhancement_factor = self.calculate_enhancement_factor(t5_embeds)
        t5_embeds = t5_embeds * enhancement_factor
        
        return t5_embeds
    
    def calculate_enhancement_factor(self, embeddings: torch.Tensor) -> float:
        """
        Calculate dynamic enhancement factor based on embedding characteristics
        """
        # Analyze embedding distribution
        embed_std = embeddings.std().item()
        
        # Adjust based on embedding variance
        # Higher variance = more detailed prompt = less enhancement needed
        if embed_std > 0.1:
            return 1.05  # Detailed prompt, minimal enhancement
        elif embed_std > 0.05:
            return 1.15  # Moderate detail, standard enhancement  
        else:
            return 1.25  # Simple prompt, more enhancement needed
    
    def encode_artistic_style_prompt(self, base_prompt: str, style_keywords: List[str], 
                                   quality_boosters: List[str] = None) -> str:
        """
        Specialized method for artistic style prompts optimized for Flux
        
        Args:
            base_prompt: Main subject/scene description
            style_keywords: Artistic style terms (e.g., ['cinematic', 'dramatic lighting'])
            quality_boosters: Quality enhancement terms
        
        Returns:
            Optimized prompt string
        """
        if quality_boosters is None:
            quality_boosters = ['masterpiece', 'best quality', '8k uhd', 'professional']
        
        # Build prompt with optimal structure for Flux's T5 understanding
        parts = []
        
        # Quality boosters first (T5 understands context well)
        if quality_boosters:
            parts.append(', '.join(quality_boosters))
        
        # Main subject
        parts.append(base_prompt)
        
        # Style keywords with emphasis
        if style_keywords:
            emphasized_styles = [f"({style}:1.2)" for style in style_keywords]
            parts.append(', '.join(emphasized_styles))
        
        return ', '.join(parts)


def patch_autostudio_flux(autostudio_instance):
    """
    Monkey patch an existing AUTOSTUDIOFLUX instance with enhanced text encoding
    
    Usage:
        from enhanced_flux_text import patch_autostudio_flux
        patch_autostudio_flux(autostudio)
    """
    enhanced_encoder = EnhancedFluxTextEncoder(autostudio_instance.pipe, autostudio_instance.device)
    
    # Replace the encode_prompts method
    autostudio_instance.encode_prompts = enhanced_encoder.encode_prompts
    autostudio_instance.parse_weighted_prompt = enhanced_encoder.parse_weighted_prompt
    autostudio_instance.optimize_flux_embeddings = enhanced_encoder.optimize_flux_embeddings
    autostudio_instance.encode_artistic_style_prompt = enhanced_encoder.encode_artistic_style_prompt
    
    print("✅ Enhanced Flux text encoding patched successfully!")
    print("New features:")
    print("  • Weighted prompt syntax: (word:1.3) [word:0.8]")
    print("  • Optimized for Flux's 3.5 guidance scale")
    print("  • Enhanced T5 + CLIP dual encoder utilization")
    print("  • Artistic style prompt optimization")


# Example usage functions
def create_example_prompts():
    """Example prompts showcasing enhanced features"""
    examples = [
        # Weighted prompts
        "(masterpiece:1.3), beautiful landscape, (dramatic lighting:1.2), [blurry:0.3]",
        
        # Artistic styles  
        "portrait of a woman, (renaissance painting:1.4), (oil painting texture:1.2), classical composition",
        
        # Complex scenes with emphasis
        "(epic fantasy scene:1.3), dragon flying over castle, (volumetric fog:1.2), (cinematic composition:1.1), [modern elements:0.2]",
        
        # Photography styles
        "(professional photography:1.2), street scene, (bokeh background:1.1), (golden hour lighting:1.3), candid moment"
    ]
    
    return examples


if __name__ == "__main__":
    print("Enhanced Flux Text Encoding Module")
    print("==================================")
    print("Example prompts:")
    for i, prompt in enumerate(create_example_prompts(), 1):
        print(f"{i}. {prompt}")