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
        Generate image using the FLUX.1-schnell model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": params
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    # Read the image data directly instead of trying to parse as JSON
                    image_data = await response.read()
                    
                    try:
                        # Try to open the image data
                        image = Image.open(io.BytesIO(image_data))
                        return image
                    except Exception as img_error:
                        raise Exception(f"Failed to process image data: {str(img_error)}")
                        
        except Exception as e:
            raise Exception(f"Image generation request failed: {str(e)}")
