# terminal.py
import os
import time
import numpy as np
import polars as pl
import typer
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.align import Align
from stable_baselines3 import PPO

# Your local imports
from core.features import SMCFeatureFactory
from core.alpaca_bridge import AlpacaQuantBridge

app = typer.Typer()
console = Console()


def generate_dashboard(price: float, equity: float, bias: float, kelly: float, log_msg: str) -> Layout:
    """Builds the Bloomberg-style terminal grid."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="market_data", ratio=1),
        Layout(name="agent_brain", ratio=1)
    )

    # Header
    header_text = f"🚀 ANTIGRAVITY QUANT TERMINAL | {datetime.utcnow().strftime('%H:%M:%S UTC')} | LIVE"
    layout["header"].update(Panel(Align.center(header_text), style="bold blue"))

    # Market & Account State
    market_table = Table(expand=True)
    market_table.add_column("Metric", style="cyan")
    market_table.add_column("Value", justify="right")
    market_table.add_row("Current Price", f"${price:,.2f}")
    market_table.add_row("Live Equity", f"${equity:,.2f}")
    layout["market_data"].update(Panel(market_table, title="[bold green]Broker State (Alpaca)"))

    # Agent Telemetry
    color = "green" if bias > 0 else "red" if bias < 0 else "yellow"
    brain_table = Table(expand=True)
    brain_table.add_column("Parameter", style="magenta")
    brain_table.add_column("Value", justify="right", style=color)
    brain_table.add_row("Directional Bias", f"{bias:+.4f}")
    brain_table.add_row("Kelly Allocation", f"{kelly:.2f}%")
    brain_table.add_row("Action", "RISK ON" if kelly > 0 else "CASH (SAFE)")
    layout["agent_brain"].update(Panel(brain_table, title="[bold magenta]PPO Agent Telemetry"))

    # Logs
    layout["footer"].update(Panel(log_msg, title="[bold white]Execution Log"))
    return layout


@app.command()
def live_trade(symbol: str = "BTC/USDT", paper: bool = True):
    console.print(f"[yellow]Booting Antigravity Engine for {symbol}...[/yellow]")

    # 1. Initialize API and Bridge
    api_key = os.environ.get("ALPACA_API_KEY", "your_key_here")
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "your_secret_here")
    bridge = AlpacaQuantBridge(api_key, sec_key, paper=paper)

    # 2. Load the winning RL Model and Feature Factory
    console.print("[cyan]Loading Production PPO Model...[/cyan]")
    model = PPO.load("models/ppo_antigrav_latest.zip")
    factory = SMCFeatureFactory()

    log_text = "System Initialized. Waiting for market data..."

    with Live(refresh_per_second=2, screen=True) as live:
        while True:
            try:
                # --- The Core Trading Loop ---
                # A. Get Data
                df = bridge.get_recent_candles(symbol, limit=150)
                current_price = df["close"].tail(1).item()
                bp, equity = bridge.get_account_metrics()

                # B. Compute Features (The RL 'Eyes')
                features_df = factory.compute_features(df)
                numeric_cols = [c for c, t in features_df.schema.items() if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
                obs_np = features_df.select(numeric_cols).tail(1).to_numpy().astype(np.float32)[0]

                # Align to 15 dimensions (Padding/Truncation) matching train.py
                if len(obs_np) < 15:
                    obs_np = np.pad(obs_np, (0, 15 - len(obs_np)), 'constant')
                elif len(obs_np) > 15:
                    obs_np = obs_np[:15]

                obs_np = np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)
                latest_obs = np.clip(obs_np, -1e3, 1e3)

                # C. Agent Inference (The RL 'Brain')
                action, _ = model.predict(latest_obs, deterministic=True)

                # Unpack action: [Bias (-1 to 1), Kelly (0 to 1)]
                raw_bias = action[0]
                kelly_raw = ((action[1] + 1) / 2)  # Scale from [-1, 1] to [0, 1]

                # D. Risk Manager Gate
                safe_kelly = kelly_raw if kelly_raw > 0.05 else 0.0

                # E. Execution
                log_text = bridge.execute_kelly_trade(symbol, raw_bias, safe_kelly)

                # Update the UI and keep clock ticking every second during the 60s polling interval
                for _ in range(60):
                    live.update(generate_dashboard(current_price, equity, raw_bias, safe_kelly * 100, log_text))
                    time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log_text = f"[red]ERROR: {str(e)}[/red]"
                live.update(generate_dashboard(0, 0, 0, 0, log_text))
                time.sleep(10)


if __name__ == "__main__":
    app()
