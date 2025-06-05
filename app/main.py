# Add at the end of the file, replace the previous launch code
if __name__ == "__main__":
    from utils.helpers import create_app
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    app = create_app(demo)
    
    if os.environ.get("RENDER"):
        # Production on Render
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            ssl_keyfile=None,
            ssl_certfile=None,
        )
    else:
        # Local development
        demo.launch(server_name="0.0.0.0", server_port=port)
