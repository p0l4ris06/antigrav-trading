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
        """Returns live buying power and equity."""
        account = self.trade_client.get_account()
        return float(account.buying_power), float(account.equity)

    def execute_kelly_trade(self, symbol: str, bias: float, kelly_fraction: float):
        """Executes a live fractional order based on the agent's Kelly output."""
        alpaca_symbol = symbol.replace("USDT", "USD")
        buying_power, _ = self.get_account_metrics()

        # Calculate exactly how many dollars to risk (Alpaca requires max 2 decimal places)
        notional_size = float(f"{float(buying_power * kelly_fraction):.2f}")

        # If Kelly is 0, or size is too small (< $2), do nothing
        if notional_size < 2.0:
            return "No Trade - Kelly below execution threshold."

        side = OrderSide.BUY if bias > 0 else OrderSide.SELL

        # Note: Shorting crypto is restricted on Alpaca depending on jurisdiction.
        # If side == SELL, you usually just close your long position.

        order_data = MarketOrderRequest(
            symbol=alpaca_symbol,
            notional=notional_size,
            side=side,
            time_in_force=TimeInForce.IOC
        )

        order = self.trade_client.submit_order(order_data=order_data)
        return f"EXECUTED: {side.name} ${notional_size:.2f} of {alpaca_symbol}"
