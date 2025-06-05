import aiohttp
from PIL import Image
import io
from typing import List, Dict
import asyncio
import time
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
                "name": "OpenJourney",
                "url": "https://api-inference.huggingface.co/models/prompthero/openjourney",
                "priority": 2,
                "strengths": ["reliable", "fast", "consistent"]
            },
            {
                "name": "FLUX",
                "url": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
                "priority": 3,
                "strengths": ["fast", "reliable"]
            },
            {
                "name": "RunwayML",
                "url": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
                "priority": 4,
                "strengths": ["consistent", "reliable"]
            },
            {
                "name": "Realistic-Vision",
                "url": "https://api-inference.huggingface.co/models/SG161222/Realistic_Vision_V5.1",
                "priority": 5,
                "strengths": ["photorealistic", "portraits"]
            }
        ]
        self.retry_delay = 1  # Initial retry delay in seconds
        self.max_retries = 3  # Maximum number of retries per model

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
            "OpenJourney": {
                "negative_prompt": f"{base_negative_prompt}, text, watermark",
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "size": 512,
            },
            "FLUX": {
                "negative_prompt": base_negative_prompt,
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "size": 768,
            },
            "RunwayML": {
                "negative_prompt": f"{base_negative_prompt}, signature, watermark",
                "num_inference_steps": 45,
                "guidance_scale": 7.5,
                "size": 512,
            },
            "Realistic-Vision": {
                "negative_prompt": f"{base_negative_prompt}, anime, cartoon, graphic",
                "num_inference_steps": 45,
                "guidance_scale": 9.0,
                "size": 768,
            }
        }
        
        default_params = model_specific_params.get(model_name, model_specific_params["SDXL-1.0"])
        
        # Update with user params but preserve negative prompt
        user_negative_prompt = user_params.get("negative_prompt", "")
        if user_negative_prompt:
            default_params["negative_prompt"] = f"{default_params['negative_prompt']}, {user_negative_prompt}"
        
        return {**default_params, **user_params}

    async def _try_generate_with_model(self, model_info: Dict, prompt: str, params: dict, attempt: int = 1) -> tuple[Image.Image, str]:
        """
        Attempt to generate image with a specific model, including retry logic
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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_info["url"], headers=headers, json=payload) as response:
                    if response.status == 402:
                        raise Exception("Rate limit exceeded")
                    elif response.status == 503:
                        if attempt <= self.max_retries:
                            await asyncio.sleep(self.retry_delay * attempt)
                            return await self._try_generate_with_model(model_info, prompt, params, attempt + 1)
                        raise Exception("Service unavailable")
                    elif response.status != 200:
                        raise Exception(f"API request failed with status {response.status}")
                    
                    image_data = await response.read()
                    
                    try:
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Enhance image quality if possible
                        if hasattr(image, 'filter'):
                            from PIL import ImageEnhance
                            enhancer = ImageEnhance.Sharpness(image)
                            image = enhancer.enhance(1.2)
                            
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(1.1)
                        
                        return image, model_info["name"]
                    except Exception as img_error:
                        raise Exception(f"Failed to process image: {str(img_error)}")
                        
        except Exception as e:
            if attempt <= self.max_retries and "Rate limit" in str(e):
                await asyncio.sleep(self.retry_delay * attempt)
                return await self._try_generate_with_model(model_info, prompt, params, attempt + 1)
            raise e

    async def generate_image(self, prompt: str, params: dict) -> tuple[Image.Image, str]:
        """
        Generate image using multiple models with fallback and retry logic
        """
        errors = []
        
        # Try models in priority order
        for model in sorted(self.models, key=lambda x: x["priority"]):
            try:
                return await self._try_generate_with_model(model, prompt, params)
            except Exception as e:
                error_msg = f"Failed with {model['name']}: {str(e)}"
                print(error_msg)  # For logging
                errors.append(error_msg)
                continue
        
        # If all models fail, raise exception with all error messages
        raise Exception(f"All image generation attempts failed:\n" + "\n".join(errors))
