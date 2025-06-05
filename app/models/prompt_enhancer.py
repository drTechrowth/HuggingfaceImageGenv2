import aiohttp
from ..config import Config

class PromptEnhancer:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.api_url = Config.HF_PROMPT_LLM

    async def enhance_prompt(self, user_intent: str) -> str:
        """
        Enhance user prompt using Zephyr-7b-beta with specific instruction tuning
        for image generation prompts.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Carefully crafted system and user prompts for Zephyr
        prompt = f"""<|system|>
You are a professional prompt engineer specialized in creating detailed, vivid prompts for AI image generation.
Your task is to enhance user prompts into detailed descriptions that will generate high-quality images.
Always include these aspects in your enhanced prompts:
- Main subject details (appearance, pose, expression)
- Environment/background details
- Lighting and atmosphere
- Artistic style
- Camera angle or perspective
- Color palette or mood

Respond ONLY with the enhanced prompt. No explanations, no additional text, no quotes.
Keep the enhanced prompt concise but detailed.</s>
<|user|>
Convert this into a detailed image generation prompt: {user_intent}</s>
<|assistant|>"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
                "stop": ["</s>", "<|user|>", "<|system|>"]
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
                        # Remove any system tokens or formatting
                        generated_text = generated_text.replace('</s>', '').strip()
                        return generated_text
                    else:
                        raise Exception("Unexpected API response format")
                        
        except Exception as e:
            raise Exception(f"Prompt enhancement failed: {str(e)}")
