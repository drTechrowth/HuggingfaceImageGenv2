import aiohttp
import base64
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                result = await response.json()
                if isinstance(result, dict) and "images" in result:
                    img_data = base64.b64decode(result["images"][0])
                else:
                    img_data = base64.b64decode(result[0])
                
                return Image.open(io.BytesIO(img_data))
