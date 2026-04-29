"""
Binance WebSocket Adapter.

Connects to Binance's public WebSocket streams and forwards L2 order book
and trade data to the Antigravity ingestion queue.

Supported streams:
    - bookTicker: real-time best bid/ask (L1)
    - trade: individual trade executions
    - depth@100ms: order book depth updates (L2)

Usage:
    # As a standalone connector
    python -m antigravity.exchange.binance --symbol BTCUSDT

    # Programmatic usage
    adapter = BinanceAdapter(symbol="BTCUSDT", queue=my_queue)
    await adapter.run()
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Binance WebSocket base URLs
BINANCE_WS_SPOT = "wss://stream.binance.com:9443/ws"
BINANCE_WS_FUTURES = "wss://fstream.binance.com/ws"


class BinanceAdapter:
    """
    Async WebSocket adapter for Binance market data.

    Subscribes to bookTicker + trade streams and normalizes
    tick data into the Antigravity TickData schema.

    Design:
        - Uses websockets library for async WS connection
        - Auto-reconnect with exponential backoff on disconnect
        - Normalizes Binance-specific field names to unified schema
        - Supports both Spot and USD-M Futures
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        queue: asyncio.Queue[dict[str, Any]] | None = None,
        futures: bool = False,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._symbol = symbol.lower()
        self._symbol_upper = symbol.upper()
        self._queue = queue
        self._futures = futures
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._running = True
        self._trade_id_counter = 0

        # Latest book state (L1)
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._bid_qty: float = 0.0
        self._ask_qty: float = 0.0

        base = BINANCE_WS_FUTURES if futures else BINANCE_WS_SPOT
        # Combined stream: bookTicker + trade
        self._url = f"{base}/{self._symbol}@bookTicker/{self._symbol}@trade"

    def stop(self) -> None:
        """Signal the adapter to stop."""
        self._running = False

    async def run(self) -> None:
        """
        Main loop: connect → receive → normalize → enqueue.
        Auto-reconnects with exponential backoff.
        """
        import websockets

        delay = self._reconnect_delay

        while self._running:
            try:
                logger.info(
                    "binance.connecting",
                    symbol=self._symbol_upper,
                    url=self._url[:60] + "...",
                    futures=self._futures,
                )

                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("binance.connected", symbol=self._symbol_upper)
                    delay = self._reconnect_delay  # reset backoff

                    async for raw_msg in ws:
                        if not self._running:
                            break

                        try:
                            msg = json.loads(raw_msg)
                            tick = self._normalize(msg)
                            if tick and self._queue:
                                try:
                                    self._queue.put_nowait(tick)
                                except asyncio.QueueFull:
                                    logger.debug("binance.queue_full")
                        except json.JSONDecodeError:
                            continue
                        except Exception as exc:
                            logger.debug("binance.parse_error", error=str(exc))

            except asyncio.CancelledError:
                logger.info("binance.cancelled")
                break
            except Exception as exc:
                logger.warning(
                    "binance.disconnected",
                    error=str(exc),
                    reconnect_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

        logger.info("binance.stopped")

    def _normalize(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """
        Normalize a Binance WebSocket message to Antigravity tick schema.

        bookTicker format:
            {"e": "bookTicker", "s": "BTCUSDT", "b": "50000.00",
             "B": "1.5", "a": "50001.00", "A": "0.8", ...}

        trade format:
            {"e": "trade", "s": "BTCUSDT", "p": "50000.50",
             "q": "0.1", "t": 123456, "T": 1700000000000, ...}
        """
        event_type = msg.get("e")

        if event_type == "bookTicker":
            self._best_bid = float(msg.get("b", 0))
            self._best_ask = float(msg.get("a", 0))
            self._bid_qty = float(msg.get("B", 0))
            self._ask_qty = float(msg.get("A", 0))

            # Emit a tick with the latest book state
            return {
                "symbol": self._symbol_upper,
                "timestamp": datetime.now(timezone.utc),
                "bid_price": self._best_bid,
                "ask_price": self._best_ask,
                "bid_size": self._bid_qty,
                "ask_size": self._ask_qty,
                "last_price": (self._best_bid + self._best_ask) / 2,
                "last_size": 0.0,
                "trade_id": 0,
            }

        elif event_type == "trade":
            price = float(msg.get("p", 0))
            qty = float(msg.get("q", 0))
            trade_id = int(msg.get("t", 0))
            trade_time_ms = int(msg.get("T", 0))

            # Convert Binance epoch ms to datetime
            ts = datetime.fromtimestamp(trade_time_ms / 1000, tz=timezone.utc)

            return {
                "symbol": self._symbol_upper,
                "timestamp": ts,
                "bid_price": self._best_bid,
                "ask_price": self._best_ask,
                "bid_size": self._bid_qty,
                "ask_size": self._ask_qty,
                "last_price": price,
                "last_size": qty,
                "trade_id": trade_id,
            }

        return None


class BinanceMultiAdapter:
    """
    Manages multiple BinanceAdapter instances for multi-symbol ingestion.

    Usage:
        adapter = BinanceMultiAdapter(
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            queue=my_queue,
        )
        await adapter.run()
    """

    def __init__(
        self,
        symbols: list[str],
        queue: asyncio.Queue[dict[str, Any]],
        futures: bool = False,
    ) -> None:
        self._adapters = [
            BinanceAdapter(symbol=s, queue=queue, futures=futures) for s in symbols
        ]

    async def run(self) -> None:
        """Run all adapters concurrently."""
        tasks = [asyncio.create_task(a.run()) for a in self._adapters]
        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Stop all adapters."""
        for a in self._adapters:
            a.stop()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
async def _run_standalone(symbol: str, futures: bool) -> None:
    """Run the adapter standalone, printing ticks to stdout."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10_000)

    adapter = BinanceAdapter(symbol=symbol, queue=queue, futures=futures)

    async def printer():
        count = 0
        while True:
            tick = await queue.get()
            count += 1
            if count % 10 == 0:  # print every 10th tick
                price = tick.get("last_price", 0)
                bid = tick.get("bid_price", 0)
                ask = tick.get("ask_price", 0)
                spread = ask - bid
                size = tick.get("last_size", 0)
                print(
                    f"[{tick['symbol']}] "
                    f"Last: {price:>10.2f} | "
                    f"Bid: {bid:>10.2f} | "
                    f"Ask: {ask:>10.2f} | "
                    f"Spread: {spread:.4f} | "
                    f"Size: {size:.4f} | "
                    f"Ticks: {count}"
                )

    await asyncio.gather(adapter.run(), printer())


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance WebSocket Adapter")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--futures", action="store_true", help="Use USD-M Futures stream")
    args = parser.parse_args()

    asyncio.run(_run_standalone(args.symbol, args.futures))


if __name__ == "__main__":
    main()
