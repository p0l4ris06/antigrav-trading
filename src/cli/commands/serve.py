import click
import uvicorn

@click.command()
@click.option("--host", default="0.0.0.0", help="Binding host")
@click.option("--port", type=int, default=8000, help="Port to run on")
def serve(host, port):
    """Start the FastAPI gateway server."""
    try:
        from antigravity.gateway.server import app
        uvicorn.run("antigravity.gateway.server:app", host=host, port=port, reload=False)
    except ImportError:
        uvicorn.run("src.antigravity.gateway.server:app", host=host, port=port, reload=False)
