import aiohttp
from PIL import Image
import io
from typing import List, Dict
import asyncio
from ..config import Config

class ImageGenerator:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        # Using smaller, faster models that are more lenient with rate limits
        self.models = [
            {
                "name": "CompVis-small",
                "url": "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4",
                "priority": 1,
            },
            {
                "name": "RPG",
                "url": "https://api-inference.huggingface.co/models/nousr/rpg",
                "priority": 2,
            },
            {
                "name": "Anything V3",
                "url": "https://api-inference.huggingface.co/models/Linaqruf/anything-v3.0",
                "priority": 3,
            }
        ]
        self.retry_delay = 2
        self.max_retries = 2

    async def _try_generate_with_model(self, model_info: Dict, prompt: str, params: dict) -> tuple[Image.Image, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Optimize parameters for faster generation and lower resource usage
        generation_params = {
            "negative_prompt": params.get("negative_prompt", "low quality, blurry"),
            "num_inference_steps": min(params.get("num_inference_steps", 30), 30),  # Cap at 30
            "guidance_scale": params.get("guidance_scale", 7.5),
            "width": 512,  # Fixed size for efficiency
            "height": 512
        }
        
        payload = {
            "inputs": prompt,
            "parameters": generation_params
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_info["url"], headers=headers, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"API request failed with status {response.status}")
                    
                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data))
                    return image, model_info["name"]
                    
        except Exception as e:
            raise Exception(f"Failed with {model_info['name']}: {str(e)}")

    async def generate_image(self, prompt: str, params: dict) -> tuple[Image.Image, str]:
        """
        Generate image using multiple models with fallback
        """
        errors = []
        
        # Try models in priority order
        for model in sorted(self.models, key=lambda x: x["priority"]):
            for attempt in range(self.max_retries):
                try:
                    return await self._try_generate_with_model(model, prompt, params)
                except Exception as e:
                    error_msg = f"Attempt {attempt + 1} failed with {model['name']}: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue
        
        raise Exception(f"All image generation attempts failed:\n" + "\n".join(errors))
