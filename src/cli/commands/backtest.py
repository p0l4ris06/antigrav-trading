import click
import os

@click.command()
@click.option("--data", "-d", help="Path to evaluation parquet data")
@click.option("--model", "-m", help="Path to trained model zip")
def backtest(data, model):
    """Run offline backtest against historical data."""
    try:
        from antigravity.rl.backtester import AntigravBacktester
    except ImportError:
        from src.antigravity.rl.backtester import AntigravBacktester
    
    backtester = AntigravBacktester()
    if model:
        backtester.load_model(model)
    else:
        backtester.load_model()
        
    if data:
        backtester.run(data)
    else:
        # Try finding a data file
        data_dir = os.getenv("AG_DATA_DIR", "data")
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
            if data_files:
                backtester.run(os.path.join(data_dir, data_files[0]))
            else:
                click.echo("No parquet files found in data directory.")
        else:
            click.echo(f"Data directory {data_dir} does not exist.")
