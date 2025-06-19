"""
Flux Text-in-Image Generation Module
Specialized prompt engineering for generating readable text within images using Flux.1
"""

from typing import List, Optional, Dict, Tuple
import re


class FluxTextGenerator:
    """
    Specialized class for generating readable text within images using Flux.1's capabilities
    """
    
    def __init__(self):
        self.text_styles = {
            'vintage': 'weathered, distressed, retro-inspired lettering',
            'modern': 'clean lines, minimalist, sans-serif typography',
            'art_deco': 'ornate, geometric patterns, decorative font',
            'handwritten': 'casual, handwritten script, natural penmanship',
            'bold_headline': 'bold, impact-style, attention-grabbing font',
            'victorian': 'ornate, Victorian-era, elaborate letterforms',
            'minimalist': 'sleek, minimalist, simple typography',
            'brushstroke': 'hand-painted, artistic, brush-style lettering',
            'neon': 'glowing, neon-style, luminous text',
            'carved': 'engraved, carved stone, 3D lettering',
            'graffiti': 'street art, spray paint, urban graffiti style',
            'corporate': 'professional, business, clean corporate font'
        }
        
        self.text_placements = {
            'center': 'centered in the image',
            'top': 'at the top of the image',
            'bottom': 'at the bottom of the image',
            'top_left': 'in the top left corner',
            'top_right': 'in the top right corner',
            'bottom_left': 'in the bottom left corner',
            'bottom_right': 'in the bottom right corner',
            'banner': 'as a banner across the image',
            'overlay': 'overlaid on the scene',
            'sign': 'on a sign within the scene',
            'background': 'integrated into the background'
        }
    
    def create_text_prompt(self, 
                          base_scene: str,
                          text_content: str,
                          text_style: str = 'modern',
                          text_color: str = 'white',
                          placement: str = 'center',
                          size: str = 'large',
                          background_context: str = None,
                          additional_effects: List[str] = None) -> str:
        """
        Create optimized prompt for Flux.1 text generation in images
        
        Args:
            base_scene: Main scene description
            text_content: The actual text to display
            text_style: Style from self.text_styles or custom description
            text_color: Color of the text
            placement: Where to place text (from self.text_placements)
            size: Size descriptor (small, medium, large, huge)
            background_context: Additional context for text integration
            additional_effects: List of effects like ['shadow', 'glow', 'outline']
        
        Returns:
            Optimized prompt string for Flux.1
        """
        
        # Get style description
        style_desc = self.text_styles.get(text_style, text_style)
        
        # Get placement description  
        placement_desc = self.text_placements.get(placement, placement)
        
        # Build text description
        text_parts = [
            f"'{text_content}'",
            f"in {style_desc}",
            f"{text_color} {size} font",
            placement_desc
        ]
        
        # Add effects if specified
        if additional_effects:
            effects_str = ', '.join(additional_effects)
            text_parts.append(f"with {effects_str}")
        
        # Combine text description
        text_description = ' '.join(text_parts)
        
        # Build final prompt with optimal structure for Flux
        prompt_parts = [
            "masterpiece, best quality, 8k uhd",  # Quality boosters
            base_scene,                           # Main scene
            f"with {text_description}"           # Text specification
        ]
        
        # Add background context if provided
        if background_context:
            prompt_parts.append(background_context)
        
        return ', '.join(prompt_parts)
    
    def create_sign_prompt(self, 
                          scene: str,
                          sign_text: str, 
                          sign_type: str = 'wooden',
                          text_style: str = 'carved') -> str:
        """Create prompt for text on signs within scenes"""
        
        return self.create_text_prompt(
            base_scene=f"{scene} with a {sign_type} sign",
            text_content=sign_text,
            text_style=text_style,
            placement='sign',
            background_context=f"sign integrated naturally into the {scene}"
        )
    
    def create_poster_prompt(self,
                           poster_style: str,
                           main_text: str,
                           subtitle: str = None,
                           color_scheme: str = 'vibrant') -> str:
        """Create prompt for poster-style images with text"""
        
        base_scene = f"{poster_style} poster design, {color_scheme} colors"
        
        if subtitle:
            text_content = f"{main_text}' with smaller subtitle '{subtitle}"
            return self.create_text_prompt(
                base_scene=base_scene,
                text_content=text_content,
                text_style='bold_headline',
                placement='center',
                size='large'
            )
        else:
            return self.create_text_prompt(
                base_scene=base_scene,
                text_content=main_text,
                text_style='bold_headline',
                placement='center',
                size='huge'
            )
    
    def create_logo_prompt(self,
                          company_name: str,
                          logo_style: str = 'modern',
                          industry: str = None) -> str:
        """Create prompt for logo generation with text"""
        
        base_scene = f"professional logo design"
        if industry:
            base_scene += f" for {industry} company"
        
        return self.create_text_prompt(
            base_scene=base_scene,
            text_content=company_name,
            text_style=logo_style,
            placement='center',
            size='large',
            background_context="clean background, professional presentation"
        )
    
    def create_book_cover_prompt(self,
                               title: str,
                               author: str,
                               genre: str,
                               mood: str = 'dramatic') -> str:
        """Create prompt for book cover with title and author"""
        
        base_scene = f"{genre} book cover, {mood} atmosphere"
        
        # Combine title and author
        text_content = f"{title}' with author name '{author}"
        
        return self.create_text_prompt(
            base_scene=base_scene,
            text_content=text_content,
            text_style='bold_headline',
            placement='center',
            size='large',
            additional_effects=['subtle shadow', 'elegant styling']
        )
    
    def optimize_for_flux_parameters(self) -> Dict[str, any]:
        """
        Return optimal Flux.1 parameters for text generation
        
        Based on research, these parameters work best for text generation
        """
        return {
            'guidance_scale': 3.0,        # Sweet spot for Flux text generation
            'num_inference_steps': 4,     # Flux schnell optimized steps
            'max_sequence_length': 512,   # Allow longer prompts for detailed text specs
            'height': 1024,
            'width': 1024
        }
    
    def create_examples(self) -> Dict[str, str]:
        """Generate example prompts for different text scenarios"""
        
        examples = {}
        
        # Restaurant sign
        examples['restaurant_sign'] = self.create_sign_prompt(
            scene="cozy Italian restaurant exterior at night",
            sign_text="Mama's Pizza",
            sign_type="vintage neon",
            text_style="neon"
        )
        
        # Movie poster
        examples['movie_poster'] = self.create_poster_prompt(
            poster_style="sci-fi thriller",
            main_text="GALAXY WARS",
            subtitle="The Final Battle",
            color_scheme="dark blue and silver"
        )
        
        # Business logo
        examples['tech_logo'] = self.create_logo_prompt(
            company_name="TechFlow",
            logo_style="modern",
            industry="technology"
        )
        
        # Book cover
        examples['fantasy_book'] = self.create_book_cover_prompt(
            title="The Dragon's Quest",
            author="J.K. Fantasy",
            genre="epic fantasy",
            mood="mystical"
        )
        
        # Street graffiti
        examples['graffiti'] = self.create_text_prompt(
            base_scene="urban brick wall",
            text_content="FREEDOM",
            text_style="graffiti",
            text_color="bright red",
            placement="center",
            size="huge",
            additional_effects=['spray paint texture', 'street art style']
        )
        
        return examples


def integrate_with_autostudio(autostudio_instance):
    """
    Add text generation capabilities to existing AutoStudio Flux instance
    """
    
    text_gen = FluxTextGenerator()
    
    # Add text generation methods to autostudio instance
    autostudio_instance.create_text_prompt = text_gen.create_text_prompt
    autostudio_instance.create_sign_prompt = text_gen.create_sign_prompt
    autostudio_instance.create_poster_prompt = text_gen.create_poster_prompt
    autostudio_instance.create_logo_prompt = text_gen.create_logo_prompt
    autostudio_instance.create_book_cover_prompt = text_gen.create_book_cover_prompt
    autostudio_instance.get_flux_text_parameters = text_gen.optimize_for_flux_parameters
    
    print("✅ Flux text generation capabilities added!")
    print("New methods available:")
    print("  • create_text_prompt() - General text in images")
    print("  • create_sign_prompt() - Text on signs")  
    print("  • create_poster_prompt() - Poster designs")
    print("  • create_logo_prompt() - Logo generation")
    print("  • create_book_cover_prompt() - Book covers")
    print("  • get_flux_text_parameters() - Optimal parameters")


if __name__ == "__main__":
    # Demo the text generation capabilities
    text_gen = FluxTextGenerator()
    
    print("🎨 Flux Text-in-Image Generation Examples")
    print("=" * 50)
    
    examples = text_gen.create_examples()
    
    for name, prompt in examples.items():
        print(f"\n📝 {name.replace('_', ' ').title()}:")
        print(f"   {prompt}")
    
    print(f"\n⚙️  Optimal Flux Parameters:")
    params = text_gen.optimize_for_flux_parameters()
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 Usage Tips:")
    print(f"   • Be extremely specific about text content and style")
    print(f"   • Use guidance_scale=3.0 for best text results")  
    print(f"   • Include placement and color details")
    print(f"   • Add effects like 'shadow' or 'glow' for better readability")