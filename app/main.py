import gradio as gr
import os
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from .models.prompt_enhancer import PromptEnhancer
from .models.image_generator import ImageGenerator
from .config import validate_env_vars

# Validate environment variables
validate_env_vars()

# Initialize models
prompt_enhancer = PromptEnhancer()
image_generator = ImageGenerator()

async def generate(user_intent: str, negative_prompt: str = None, num_inference_steps: int = 30, guidance_scale: float = 7.5) -> tuple:
    """
    Generate image from user prompt with enhanced prompt engineering.
    Falls back to original prompt if enhancement fails.
    """
    try:
        # Create progress
        progress = gr.Progress()
        
        # Try to enhance the prompt
        try:
            progress(0.2, desc="Enhancing prompt...")
            enhanced_prompt = await prompt_enhancer.enhance_prompt(user_intent)
            prompt_to_use = enhanced_prompt
            progress(0.3, desc="Prompt enhanced successfully!")
        except Exception as prompt_error:
            # If prompt enhancement fails, use the original prompt
            progress(0.3, desc="Prompt enhancement failed, using original prompt...")
            prompt_to_use = user_intent
            print(f"Prompt enhancement failed: {str(prompt_error)}")  # For logging
        
        # Generate image with either enhanced or original prompt
        progress(0.4, desc="Generating image...")
        generation_params = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        image = await image_generator.generate_image(prompt_to_use, generation_params)
        
        # Complete progress
        progress(1.0, desc="Complete!")
        
        # Return both the image and the prompt that was actually used
        return image, prompt_to_use
            
    except Exception as e:
        raise gr.Error(f"Image generation failed: {str(e)}")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Image Generator with Enhanced Prompts")
    
    with gr.Row():
        with gr.Column():
            user_intent = gr.Textbox(
                label="Describe your image idea",
                placeholder="Example: A serene lake at sunset with mountains in the background",
                lines=3
            )
            enhanced_prompt = gr.Textbox(
                label="Used Prompt (Enhanced or Original)",
                interactive=False,
                info="If prompt enhancement fails, the original prompt will be used"
            )
            
        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            
    with gr.Row():
        generate_btn = gr.Button("Generate Image", variant="primary")
        clear_btn = gr.Button("Clear")
        
    with gr.Accordion("Advanced Options", open=False):
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            placeholder="What to avoid in the image"
        )
        with gr.Row():
            num_inference_steps = gr.Slider(
                minimum=20, maximum=100, value=30,
                label="Number of Steps"
            )
            guidance_scale = gr.Slider(
                minimum=1, maximum=20, value=7.5,
                label="Guidance Scale"
            )
            
    generate_btn.click(
        fn=generate,
        inputs=[user_intent, negative_prompt, num_inference_steps, guidance_scale],
        outputs=[output_image, enhanced_prompt]
    )
    clear_btn.click(
        fn=lambda: (None, None),
        inputs=[],
        outputs=[output_image, enhanced_prompt]
    )

    gr.Markdown("""
    ### Notes:
    - If prompt enhancement fails, the system will automatically use your original prompt
    - The "Used Prompt" field shows which prompt was actually used for generation
    """)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    if os.environ.get("RENDER"):
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    else:
        demo.launch(server_name="0.0.0.0", server_port=port)
