import aiohttp
from PIL import Image
import io
from typing import List, Dict
from ..config import Config

class ImageGenerator:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.models = [
            {
                "name": "SDXL-1.0",
                "url": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-1.0",
                "priority": 1,
                "strengths": ["photorealistic", "high-quality", "detailed"]
            },
            {
                "name": "FLUX",
                "url": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                "priority": 2,
                "strengths": ["fast", "reliable"]
            },
            {
                "name": "Realistic-Vision",
                "url": "https://api-inference.huggingface.co/models/SG161222/Realistic_Vision_V5.1",
                "priority": 3,
                "strengths": ["photorealistic", "portraits"]
            },
            {
                "name": "PhotoReal",
                "url": "https://api-inference.huggingface.co/models/fashioniq/photoreal-xl",
                "priority": 4,
                "strengths": ["photorealistic", "commercial"]
            }
        ]

    def _get_optimal_parameters(self, model_name: str, user_params: dict) -> dict:
        """
        Get optimized parameters for specific models
        """
        base_negative_prompt = "cartoon, anime, illustration, painting, drawing, artwork, graphic, unrealistic, low quality, blurry, distorted"
        
        model_specific_params = {
            "SDXL-1.0": {
                "negative_prompt": base_negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 8.5,
                "size": 1024,
            },
            "FLUX": {
                "negative_prompt": base_negative_prompt,
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "size": 768,
            },
            "Realistic-Vision": {
                "negative_prompt": f"{base_negative_prompt}, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch",
                "num_inference_steps": 45,
                "guidance_scale": 9.0,
                "size": 768,
            },
            "PhotoReal": {
                "negative_prompt": f"{base_negative_prompt}, watermark, text, logo, signature",
                "num_inference_steps": 45,
                "guidance_scale": 8.0,
                "size": 896,
            }
        }
        
        # Get default params for the model
        default_params = model_specific_params.get(model_name, model_specific_params["SDXL-1.0"])
        
        # Update with user params but preserve negative prompt
        user_negative_prompt = user_params.get("negative_prompt", "")
        if user_negative_prompt:
            default_params["negative_prompt"] = f"{default_params['negative_prompt']}, {user_negative_prompt}"
        
        # Merge default params with user params
        return {**default_params, **user_params}

    async def _try_generate_with_model(self, model_info: Dict, prompt: str, params: dict) -> tuple[Image.Image, str]:
        """
        Attempt to generate image with a specific model
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        generation_params = self._get_optimal_parameters(model_info["name"], params)
        
        payload = {
            "inputs": prompt,
            "parameters": generation_params
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model_info["url"], headers=headers, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}")
                
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Enhance image quality
                if hasattr(image, 'filter'):
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(1.2)
                    
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.1)
                
                return image, model_info["name"]

    async def generate_image(self, prompt: str, params: dict) -> tuple[Image.Image, str]:
        """
        Generate image using multiple models with fallback
        """
        errors = []
        
        # Try models in priority order
        for model in sorted(self.models, key=lambda x: x["priority"]):
            try:
                image, model_name = await self._try_generate_with_model(model, prompt, params)
                print(f"Successfully generated image using {model_name}")
                return image, model_name
            except Exception as e:
                error_msg = f"Failed with {model['name']}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        # If all models fail, raise exception with all error messages
        raise Exception(f"All image generation attempts failed:\n" + "\n".join(errors))
