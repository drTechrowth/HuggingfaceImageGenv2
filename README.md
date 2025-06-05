# HuggingFace Image Generator

An AI-powered image generation application using HuggingFace's FLUX.1-schnell model with enhanced prompt engineering.

## Features

- Enhanced prompt engineering using specialized LLM
- High-quality image generation with FLUX.1-schnell
- User-friendly interface with Gradio
- Advanced options for fine-tuning generation
- Asynchronous processing for better performance

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/drTechrowth/HugginfaceimageGen.git
cd HugginfaceimageGen
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your HuggingFace API key
```

5. Run the application:
```bash
python -m app.main
```

## Deployment on Render

1. Fork this repository to your GitHub account.

2. Create a new Web Service on Render:
   - Connect your GitHub repository
   - Choose Python environment
   - The build command and start command are already configured in `render.yaml`

3. Set up environment variables:
   - Go to your Web Service Dashboard
   - Add the following environment variables:
     - `HF_API_KEY`: Your HuggingFace API key
     - Other variables are configured in render.yaml

4. Deploy:
   - Render will automatically deploy your application
   - Any push to the main branch will trigger a new deployment

## Environment Variables

- `HF_API_KEY`: Your HuggingFace API key (required)
- `HF_API_TTI_BASE`: FLUX.1-schnell model endpoint (configured in render.yaml)
- `HF_PROMPT_LLM`: Prompt enhancement model endpoint (configured in render.yaml)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
