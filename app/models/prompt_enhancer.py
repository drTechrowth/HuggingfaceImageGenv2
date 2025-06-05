import aiohttp
from ..config import Config

class PromptEnhancer:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.api_url = Config.HF_PROMPT_LLM

    async def enhance_prompt(self, user_intent: str) -> str:
        """
        Enhance user prompt using specialized LLM for better image generation results.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are an expert at writing prompts for image generation. 
        Convert the user's input into a detailed, vivid prompt that will generate high-quality images.
        Focus on descriptive details, artistic style, lighting, atmosphere, and composition.
        Keep the output concise but detailed."""

        payload = {
            "inputs": f"{system_prompt}\n\nUser: {user_intent}\nAssistant: Generate a detailed image prompt for: {user_intent}\n",
            "parameters": {
                "max_new_tokens": 150,
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
                    
                    if response.content_type == 'application/json':
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0]['generated_text'].strip()
                        else:
                            raise Exception("Unexpected API response format")
                    else:
                        raise Exception(f"Unexpected content type: {response.content_type}")
        except Exception as e:
            raise Exception(f"Prompt enhancement failed: {str(e)}")
