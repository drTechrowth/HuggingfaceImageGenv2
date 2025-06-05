from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import gradio as gr

def add_health_check(app: FastAPI):
    """
    Add health check endpoint for Render
    """
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

def setup_cors(app: FastAPI):
    """
    Setup CORS for Render deployment
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def create_app(demo: gr.Blocks) -> FastAPI:
    """
    Create FastAPI app with Gradio and health check
    """
    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    add_health_check(app)
    setup_cors(app)
    return app
