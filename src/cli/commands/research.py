import click
import sys

@click.group()
def research():
    """LLM-driven quantitative research optimization."""
    pass

@research.command("optimize")
@click.option("--provider", default="ollama", help="LLM Provider: ollama | azure | anthropic | gemini")
@click.option("--model", default="deepseek-coder-v2", help="LLM Model name")
@click.option("--iterations", type=int, default=50, help="Number of research iterations")
@click.option("--patience", type=int, default=8, help="Patience before stopping research")
def optimize(provider, model, iterations, patience):
    """Run the auto-optimizer research loop."""
    try:
        from antigravity.research.mutator import main as run_mutator
    except ImportError:
        from src.antigravity.research.mutator import main as run_mutator
    
    args = ["research", "--provider", provider, "--model", model, "--iterations", str(iterations), "--patience", str(patience)]
    sys.argv = args
    run_mutator()
