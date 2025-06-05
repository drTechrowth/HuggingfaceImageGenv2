import aiohttp
from ..config import Config

class PromptEnhancer:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.api_url = Config.HF_PROMPT_LLM

    async def enhance_prompt(self, user_intent: str) -> str:
        """
        Enhance user prompt for photorealistic image generation.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Specialized prompt engineering for photorealistic results
        system_prompt = """<|system|>
You are an expert at crafting prompts specifically for photorealistic image generation.
Your task is to enhance user prompts to generate highly realistic, photograph-like images.

Follow these rules for photorealistic prompts:
1. Always include photography-specific terms (e.g., depth of field, focal length, lighting conditions)
2. Specify camera details (e.g., DSLR, high-resolution, 8K, RAW format)
3. Add realistic lighting descriptions (e.g., golden hour, studio lighting, natural sunlight)
4. Include environment details that ground the image in reality
5. Use specific materials and textures that exist in the real world
6. Add photographic composition elements (rule of thirds, leading lines)
7. Specify time of day and weather conditions when relevant
8. Include subtle imperfections for realism (slight grain, natural shadows)

AVOID:
- Cartoon or artistic style terms
- Fantasy or unrealistic elements
- Abstract concepts
- Vague descriptions

FORMAT: Translate the user's intention into a detailed photographic prompt
WITHOUT any explanations or additional text.</s>
<|user|>
Create a photorealistic prompt for: {user_intent}</s>
<|assistant|>"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        # Clean up the response
                        generated_text = result[0].get('generated_text', '').strip()
                        # Add photorealistic quality markers if not present
                        if "photorealistic" not in generated_text.lower():
                            generated_text = f"photorealistic, highly detailed, 8K UHD, DSLR photograph, {generated_text}"
                        return generated_text
                    else:
                        raise Exception("Unexpected API response format")
                        
        except Exception as e:
            raise Exception(f"Prompt enhancement failed: {str(e)}")
