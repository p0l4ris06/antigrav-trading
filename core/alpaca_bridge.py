# core/alpaca_bridge.py
import polars as pl
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class AlpacaQuantBridge:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.data_client = CryptoHistoricalDataClient(api_key, secret_key)
        self.trade_client = TradingClient(api_key, secret_key, paper=paper)

    def get_recent_candles(self, symbol: str, limit: int = 150) -> pl.DataFrame:
        """Fetches the last N 15-minute candles and formats them for features.py"""
        # Alpaca uses BTC/USD, standard crypto uses BTC/USDT
        alpaca_symbol = symbol.replace("USDT", "USD")

        request_params = CryptoBarsRequest(
            symbol_or_symbols=[alpaca_symbol],
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=datetime.utcnow() - timedelta(days=4)  # Ensure enough data for 150 bars
        )
        bars = self.data_client.get_crypto_bars(request_params).df

        if bars.empty:
            raise ValueError(f"No data returned from Alpaca for {symbol}")

        # Reset pandas index and convert to Polars
        bars = bars.reset_index()
        df = pl.from_pandas(bars)

        # Rename and cast to exactly match what SMCFeatureFactory expects
        df = df.select([
            pl.col("timestamp").alias("timestamp"),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
        ]).tail(limit)

        return df

    def get_account_metrics(self):
        """Returns live cash (buying power for crypto) and equity."""
        account = self.trade_client.get_account()
        return float(account.cash), float(account.equity)

    def execute_kelly_trade(self, symbol: str, bias: float, kelly_fraction: float):
        alpaca_symbol = symbol.replace("USDT", "USD")

        # 1. Check current positions
        try:
            position = self.trade_client.get_open_position(alpaca_symbol)
            current_qty = float(position.qty)
        except Exception:
            current_qty = 0.0

        buying_power, equity = self.get_account_metrics()
        # Cap order size to actual available cash balance to prevent 403 insufficient funds
        max_possible = min(equity * kelly_fraction, buying_power)
        target_notional = float(f"{float(max_possible):.2f}")

        # If Kelly is 0 (Agent wants cash)
        if kelly_fraction == 0.0:
            if current_qty > 0:
                # Close the long position
                self.trade_client.close_position(alpaca_symbol)
                return f"CLOSED LONG POSITION - Returning to Cash."
            return "Holding Cash."

        # If Kelly > 0 (Agent wants to be Long)
        # Note: We only allow Longs now based on the terminal.py logic
        if target_notional < 2.0:
            return f"No Trade - Available USD Balance (${buying_power:.2f}) below $2.00 threshold."

        # Simplistic Execution: For now, just buy the target notional.
        # (A real system would calculate the delta between current position and target)
        if current_qty == 0:
            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                notional=target_notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.IOC
            )
            self.trade_client.submit_order(order_data=order_data)
            return f"EXECUTED: BUY ${target_notional:.2f} of {alpaca_symbol}"

        return f"Holding current LONG position."
