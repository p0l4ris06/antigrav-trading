import click
import sys

@click.group()
def data():
    """Manage the data pipeline (harvesting, validation)."""
    pass

@data.command("harvest")
@click.option("--symbols", "-s", multiple=True, help="Crypto symbols to harvest (e.g. BTC/USDT)")
@click.option("--days", type=int, default=365, help="Number of days of history to pull")
def harvest(symbols, days):
    """Harvest OHLCV data from exchanges."""
    try:
        from antigravity.data.harvester import main as run_harvester
    except ImportError:
        from src.antigravity.data.harvester import main as run_harvester
    
    args = ["data", "--days", str(days)]
    if symbols:
        args.append("--crypto")
        args.extend(symbols)
    
    sys.argv = args
    run_harvester()
