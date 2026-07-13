import click
import sys

@click.command()
@click.option("--data", "-d", multiple=True, help="Path to data file(s) or directories")
@click.option("--timesteps", type=int, default=10000, help="Number of training timesteps")
@click.option("--agent-type", default="ppo", help="RL agent algorithm type")
def train(data, timesteps, agent_type):
    """Train the Kelly-Convex RL trading agent."""
    try:
        from antigravity.rl.training import main as run_train
    except ImportError:
        from src.antigravity.rl.training import main as run_train

    # Construct arguments list for train.py's parser
    args = ["train"]
    if data:
        args.append("--data")
        args.extend(data)
    args.extend(["--timesteps", str(timesteps)])
    args.extend(["--agent-type", agent_type])

    sys.argv = args
    run_train()
