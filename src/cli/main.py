import click
import sys
import os

# Ensure src/ is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@click.group()
def app():
    """Antigravity Trading System Command Line Interface."""
    pass

try:
    from cli.commands.train import train
    from cli.commands.live import live
    from cli.commands.research import research
    from cli.commands.data import data
    from cli.commands.serve import serve
    from cli.commands.backtest import backtest
    from cli.commands.model import model
    from cli.commands.terminal import terminal
except ImportError:
    from src.cli.commands.train import train
    from src.cli.commands.live import live
    from src.cli.commands.research import research
    from src.cli.commands.data import data
    from src.cli.commands.serve import serve
    from src.cli.commands.backtest import backtest
    from src.cli.commands.model import model
    from src.cli.commands.terminal import terminal

app.add_command(train)
app.add_command(live)
app.add_command(research)
app.add_command(data)
app.add_command(serve)
app.add_command(backtest)
app.add_command(model)
app.add_command(terminal)

if __name__ == "__main__":
    app()
