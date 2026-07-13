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
from antigravity.features.base import SMCFeatureFactory
from antigravity.data.sources import AlpacaQuantBridge
import click

console = Console()


def generate_dashboard(price: float, equity: float, bias: float, kelly: float, action_text: str, log_msg: str) -> Layout:
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
    brain_table.add_row("Action", action_text)
    layout["agent_brain"].update(Panel(brain_table, title="[bold magenta]PPO Agent Telemetry"))

    # Logs
    layout["footer"].update(Panel(log_msg, title="[bold white]Execution Log"))
    return layout


@click.command()
@click.option("--symbol", default="BTC/USDT", help="Symbol to trade")
@click.option("--paper", is_flag=True, default=True, help="Run in paper trading mode")
def terminal(symbol, paper):
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

    # ── Circuit Breaker State ──────────────────────────────────────────────────
    # These constants are INDEPENDENT of the PPO agent output.
    # They cap safe_kelly BEFORE it reaches execute_kelly_trade().
    MAX_ORDER_NOTIONAL = 5_000.0   # Hard cap: no single order may exceed $5,000 notional
    MAX_DAILY_DD = 0.05            # Kill switch: halt if equity drops 5% from session HWM
    _session_equity_hwm: float = 0.0   # High-water mark initialised on first successful tick
    _trading_halted: bool = False      # Latches True on drawdown breach; requires restart to clear
    # ─────────────────────────────────────────────────────────────────────────

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
                exclude_cols = {"open", "high", "low", "close", "volume", "true_range"}
                numeric_cols = [c for c, t in features_df.schema.items() if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and c not in exclude_cols]
                obs_np = features_df.select(numeric_cols).tail(1).to_numpy().astype(np.float32)[0]

                # Align to 9 dimensions (Padding/Truncation) matching train.py
                if len(obs_np) < 9:
                    obs_np = np.pad(obs_np, (0, 9 - len(obs_np)), 'constant')
                elif len(obs_np) > 9:
                    obs_np = obs_np[:9]

                obs_np = np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)
                latest_obs = np.clip(obs_np, -1e3, 1e3)

                # C. Agent Inference (The RL 'Brain')
                vec_normalize_path = "models/vec_normalize.pkl"
                if os.path.exists(vec_normalize_path):
                    import pickle
                    try:
                        with open(vec_normalize_path, "rb") as f:
                            vec_norm = pickle.load(f)
                        vec_norm.training = False
                        vec_norm.norm_reward = False
                        
                        obs_batched = np.expand_dims(latest_obs, axis=0)
                        obs_normalized = vec_norm.normalize_obs(obs_batched)
                        action, _ = model.predict(obs_normalized, deterministic=True)
                        action = action[0]
                    except Exception as e:
                        action, _ = model.predict(latest_obs, deterministic=True)
                else:
                    action, _ = model.predict(latest_obs, deterministic=True)

                # Assuming your model outputs 2 continuous values: [Bias, Kelly]
                raw_bias = float(action[0])
                # Scale Kelly from [-1, 1] to [0, 1]
                kelly_raw = float(((action[1] + 1) / 2)) if len(action) > 1 else abs(raw_bias)

                # D. Strict Long-Only Risk Manager
                # If bias is negative (Short), force Kelly to 0 (Cash)
                if raw_bias <= 0:
                    safe_kelly = 0.0
                    action_text = "CASH (BEARISH BIAS)"
                else:
                    # If Long, require at least 20% confidence to beat the spread
                    safe_kelly = kelly_raw if kelly_raw > 0.20 else 0.0
                    action_text = "BUY (LONG)" if safe_kelly > 0 else "CASH (LOW CONFIDENCE)"

                # ── Independent Circuit Breakers ──────────────────────────────
                # Evaluated AFTER feature scaling and AFTER the agent produces
                # its raw action — these guards are purely reactive, never
                # influencing what the model sees.

                # 1. Session HWM initialisation (first tick only)
                if _session_equity_hwm == 0.0 and equity > 0:
                    _session_equity_hwm = equity

                # 2. Daily drawdown kill switch
                if equity > _session_equity_hwm:
                    _session_equity_hwm = equity   # update HWM on new highs
                if _session_equity_hwm > 0 and equity < _session_equity_hwm * (1.0 - MAX_DAILY_DD):
                    dd_pct = (_session_equity_hwm - equity) / _session_equity_hwm * 100
                    if not _trading_halted:
                        log_text = (f"[bold red]CIRCUIT BREAKER: Daily drawdown {dd_pct:.1f}% ≥ {MAX_DAILY_DD*100:.0f}% limit. "
                                    f"Trading HALTED. Restart terminal to resume.[/bold red]")
                        _trading_halted = True
                    safe_kelly = 0.0
                    action_text = "HALTED (MAX DD)"

                # 3. Hard max-notional cap
                #    kelly_fraction × equity must not exceed MAX_ORDER_NOTIONAL.
                #    Caps the fraction, not the execution — execute_kelly_trade
                #    receives a valid [0, 1] fraction and applies it to equity.
                if safe_kelly > 0 and equity > 0:
                    kelly_notional = safe_kelly * equity
                    if kelly_notional > MAX_ORDER_NOTIONAL:
                        safe_kelly = MAX_ORDER_NOTIONAL / equity   # back-calculate capped fraction
                        action_text = f"BUY (NOTIONAL CAPPED @ ${MAX_ORDER_NOTIONAL:,.0f})"
                # ─────────────────────────────────────────────────────────────

                # E. Execution
                log_text = bridge.execute_kelly_trade(symbol, raw_bias, safe_kelly)

                # Update the UI and keep clock ticking every second during the 60s polling interval
                for _ in range(60):
                    live.update(generate_dashboard(current_price, equity, raw_bias, safe_kelly * 100, action_text, log_text))
                    time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log_text = f"[red]ERROR: {str(e)}[/red]"
                live.update(generate_dashboard(0, 0, 0, 0, "ERROR", log_text))
                time.sleep(10)


if __name__ == "__main__":
    terminal()
