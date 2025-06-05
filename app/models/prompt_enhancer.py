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
        
        # Llama-2 specific prompt format
        prompt = f"""<s>[INST] You are an expert at crafting prompts for AI image generation.
        Convert this user request into a detailed, vivid prompt that will generate high-quality images.
        Focus on descriptive details, artistic style, lighting, atmosphere, and composition.
        Keep the output concise but detailed.

        User Request: {user_intent}

        Generate only the enhanced prompt, no explanations or additional text. [/INST]"""
        
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
                        # Remove any Llama-2 specific tokens and clean up the response
                        generated_text = result[0].get('generated_text', '').strip()
                        generated_text = generated_text.replace('</s>', '').strip()
                        return generated_text
                    else:
                        raise Exception("Unexpected API response format")
                        
        except Exception as e:
            raise Exception(f"Prompt enhancement failed: {str(e)}")
