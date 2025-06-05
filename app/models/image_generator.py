import aiohttp
from PIL import Image
import io
from ..config import Config

class ImageGenerator:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.api_url = Config.HF_API_TTI_BASE

    async def generate_image(self, prompt: str, params: dict) -> Image.Image:
        """
        Generate photorealistic image using optimized parameters.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Optimize parameters for photorealism
        default_params = {
            "negative_prompt": "cartoon, anime, illustration, painting, drawing, artwork, graphic, unrealistic, low quality, blurry, distorted",
            "num_inference_steps": 50,  # Increased for better quality
            "guidance_scale": 8.5,      # Adjusted for more photorealism
            "size": 1024,              # Larger size for more detail
        }
        
        # Update with user params but preserve negative prompt
        user_negative_prompt = params.get("negative_prompt", "")
        if user_negative_prompt:
            default_params["negative_prompt"] = f"{default_params['negative_prompt']}, {user_negative_prompt}"
        
        # Merge default params with user params
        generation_params = {**default_params, **params}
        
        payload = {
            "inputs": prompt,
            "parameters": generation_params
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    # Read the image data directly
                    image_data = await response.read()
                    
                    try:
                        # Process the image
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Optional: Enhance image quality
                        if hasattr(image, 'filter'):
                            from PIL import ImageEnhance
                            enhancer = ImageEnhance.Sharpness(image)
                            image = enhancer.enhance(1.2)  # Slight sharpness enhancement
                            
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(1.1)  # Slight contrast enhancement
                        
                        return image
                    except Exception as img_error:
                        raise Exception(f"Failed to process image data: {str(img_error)}")
                        
        except Exception as e:
            raise Exception(f"Image generation request failed: {str(e)}")
