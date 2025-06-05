import aiohttp
import os
from ..config import Config

class PromptEnhancer:
    def __init__(self):
        self.api_key = Config.HF_API_KEY
        self.api_url = Config.HF_PROMPT_LLM

    async def enhance_prompt(self, user_intent: str) -> str:
        """
        Enhance user prompt using specialized LLM for better image generation results.
        """
        system_prompt = """
        You are an expert at crafting prompts for image generation. 
        Transform the user's input into a detailed, vivid prompt that will generate high-quality images.
        Focus on:
        - Descriptive details
        - Artistic style
        - Lighting and atmosphere
        - Composition
        - Technical aspects (quality, resolution)
        Do not include NSFW content or copyrighted characters.
        """
        
        user_message = f"Transform this into a detailed image generation prompt: {user_intent}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as response:
                result = await response.json()
                return result[0]["generated_text"].strip()
            
