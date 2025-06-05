import torch
from diffusers import (
    StableDiffusionPipeline, 
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline
)
from PIL import Image
import io
from typing import List, Dict
import os
from ..config import Config

class ImageGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.models = [
            {
                "name": "CompVis/stable-diffusion-v1-4",
                "pipeline_class": StableDiffusionPipeline,
                "priority": 1,
                "strengths": ["fast", "reliable", "local"]
            },
            {
                "name": "runwayml/stable-diffusion-v1-5",
                "pipeline_class": StableDiffusionPipeline,
                "priority": 2,
                "strengths": ["quality", "reliable"]
            }
        ]
        
        self.loaded_models = {}
        self._initialize_primary_model()

    def _initialize_primary_model(self):
        """Initialize the primary model at startup"""
        primary_model = sorted(self.models, key=lambda x: x["priority"])[0]
        self._load_model(primary_model["name"], primary_model["pipeline_class"])

    def _load_model(self, model_id: str, pipeline_class) -> any:
        """Load a model if not already loaded"""
        if model_id not in self.loaded_models:
            try:
                pipeline = pipeline_class.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    local_files_only=True  # Try local first
                )
                if self.device == "cuda":
                    pipeline = pipeline.to(self.device)
                # Use DPM++ scheduler for better quality
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config
                )
                self.loaded_models[model_id] = pipeline
            except Exception as e:
                print(f"Failed to load model {model_id} locally, downloading: {str(e)}")
                # If local load fails, try downloading
                pipeline = pipeline_class.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None
                )
                if self.device == "cuda":
                    pipeline = pipeline.to(self.device)
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config
                )
                self.loaded_models[model_id] = pipeline
        return self.loaded_models[model_id]

    def _get_optimal_parameters(self, model_name: str, user_params: dict) -> dict:
        """Get optimized parameters for specific models"""
        base_negative_prompt = (
            "cartoon, anime, illustration, painting, drawing, artwork, graphic, "
            "unrealistic, low quality, blurry, distorted, watermark, signature, text"
        )
        
        default_params = {
            "negative_prompt": base_negative_prompt,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
        
        # Update with user params but preserve negative prompt
        user_negative_prompt = user_params.get("negative_prompt", "")
        if user_negative_prompt:
            default_params["negative_prompt"] = f"{default_params['negative_prompt']}, {user_negative_prompt}"
        
        return {**default_params, **user_params}

    async def generate_image(self, prompt: str, params: dict) -> tuple[Image.Image, str]:
        """Generate image using local models"""
        errors = []
        
        for model in sorted(self.models, key=lambda x: x["priority"]):
            try:
                # Load model if not already loaded
                pipeline = self._load_model(model["name"], model["pipeline_class"])
                
                # Get optimized parameters
                generation_params = self._get_optimal_parameters(model["name"], params)
                
                # Generate image
                with torch.no_grad():
                    output = pipeline(
                        prompt=prompt,
                        **generation_params
                    )
                
                image = output.images[0]
                
                # Optional image enhancement
                if hasattr(image, 'filter'):
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(1.2)
                    
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.1)
                
                return image, f"{model['name']} (Local)"
                
            except Exception as e:
                error_msg = f"Failed with {model['name']}: {str(e)}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        raise Exception(f"All image generation attempts failed:\n" + "\n".join(errors))
