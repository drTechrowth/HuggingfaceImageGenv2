import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def validate_env_vars():
    required_vars = ["HF_API_KEY", "HF_API_TTI_BASE", "HF_PROMPT_LLM"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configuration class
class Config:
    HF_API_KEY = os.getenv("HF_API_KEY")
    HF_API_TTI_BASE = os.getenv("HF_API_TTI_BASE", "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell")
    HF_PROMPT_LLM = os.getenv("HF_PROMPT_LLM", "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta")
