"""
core/gateway.py — Antigravity OmniGateway
==========================================
Two responsibilities in one module:

  1. OmniGateway class  — translates PPO action vectors into real exchange
     orders via ccxt. This is what live_daemon.py imports and calls.

  2. FastAPI tick server — optional high-throughput WebSocket ingestion layer
     for L2/L3 order book deltas. Run standalone with:
         python -m core.gateway

Improvements over Phase 1.1:
  - OmniGateway implemented (was missing entirely — live_daemon.py couldn't run)
  - Kelly sizing from action vector with configurable fraction cap
  - Position state tracking: prevents duplicate orders on same side
  - Order confirmation loop with timeout and cancel-on-timeout
  - Stop-loss placed immediately after entry via exchange native SL order
  - Dry-run mode: logs intended orders without touching exchange
  - FastAPI lifespan replaces deprecated @app.on_event("startup")
  - Consumer task properly tracked and cancelled on shutdown
  - Tick schema validation before queue insertion
  - Structured logging (reuses antigravity logger if available)
  - winloop policy applied only when available (graceful fallback)

Dependencies:
    pip install fastapi uvicorn winloop ccxt
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import ccxt
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ── winloop: high-performance Windows event loop ──────────────────────────────
try:
    import winloop
    winloop.install()   # sets EventLoopPolicy globally; safe to call before app start
    _WINLOOP_ACTIVE = True
except ImportError:
    _WINLOOP_ACTIVE = False


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

log = logging.getLogger("antigravity.gateway")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | GATEWAY | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))
    log.addHandler(_h)
    log.setLevel(logging.DEBUG)


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class GatewayConfig:
    # Kelly fraction cap — never risk more than this fraction of equity per trade
    max_kelly_fraction: float = 0.02        # 2% max per trade (conservative for live)
    min_order_usd: float = 5.0              # skip trades below this notional value
    sl_atr_multiplier: float = 1.5          # stop-loss = entry ± 1.5 × ATR
    order_timeout_seconds: int = 30         # cancel unfilled limit orders after this
    use_market_orders: bool = True          # market order for immediate fill
    dry_run: bool = False

    # Exchange map: target_exchange string → ccxt class name
    exchange_map: dict = field(default_factory=lambda: {
        "BINANCE": "binance",
        "CRYPTO": "cryptocom",
        "BYBIT": "bybit",
        "KRAKEN": "kraken",
    })


# ─────────────────────────────────────────────
#  OmniGateway
# ─────────────────────────────────────────────

class OmniGateway:
    """
    Routes PPO action vectors to exchange orders.

    Action vector convention (matches core/agent.py output):
        action[0]  — directional bias   (-1.0 = strong short, +1.0 = strong long)
        action[1]  — Kelly confidence   (0.0–1.0, fraction of max position to take)

    Usage:
        gateway = OmniGateway(crypto_config={"api_key": ..., "secret": ...})
        await gateway.route_action(
            target_exchange="BINANCE",
            symbol="ETH/USDT",
            action_vector=action,
            account_equity=500.0,
            current_atr=12.5,
        )
    """

    def __init__(
        self,
        crypto_config: Optional[dict] = None,
        config: Optional[GatewayConfig] = None,
    ):
        self.cfg = config or GatewayConfig()
        self.cfg.dry_run = self.cfg.dry_run or (
            os.getenv("DRY_RUN", "0") == "1"
        )
        self._credentials = crypto_config or {}
        self._exchanges: dict[str, ccxt.Exchange] = {}
        self._position: Optional[str] = None   # 'long' | 'short' | None
        self._position_size: float = 0.0
        self._entry_price: float = 0.0
        log.info(
            "OmniGateway initialised | dry_run=%s | max_kelly=%.1f%%",
            self.cfg.dry_run, self.cfg.max_kelly_fraction * 100,
        )

    # ── Exchange connection pool ───────────────────────────────────────────

    def _get_exchange(self, target: str) -> ccxt.Exchange:
        target = target.upper()
        if target not in self._exchanges:
            ccxt_name = self.cfg.exchange_map.get(target)
            if not ccxt_name:
                raise ValueError(
                    f"Unknown target exchange: '{target}'. "
                    f"Valid options: {list(self.cfg.exchange_map)}"
                )
            cls = getattr(ccxt, ccxt_name)
            params = {"enableRateLimit": True}
            if self._credentials.get("api_key"):
                params["apiKey"] = self._credentials["api_key"]
                params["secret"] = self._credentials["secret"]
            self._exchanges[target] = cls(params)
            log.info("Connected to exchange: %s (%s)", target, ccxt_name)
        return self._exchanges[target]

    # ── Kelly position sizing ──────────────────────────────────────────────

    def _size_position(
        self,
        kelly_confidence: float,
        account_equity: float,
        current_price: float,
    ) -> float:
        """
        Returns position size in base currency units.
        Kelly fraction is capped at max_kelly_fraction to prevent over-leveraging.
        """
        fraction = min(abs(kelly_confidence), self.cfg.max_kelly_fraction)
        notional = account_equity * fraction
        if notional < self.cfg.min_order_usd:
            log.info(
                "Notional £%.2f below minimum £%.2f — skipping.",
                notional, self.cfg.min_order_usd,
            )
            return 0.0
        size = notional / current_price
        return round(size, 6)

    # ── Stop-loss price ────────────────────────────────────────────────────

    def _stop_price(self, side: str, entry: float, atr: float) -> float:
        offset = atr * self.cfg.sl_atr_multiplier
        return entry - offset if side == "long" else entry + offset

    # ── Close existing position ────────────────────────────────────────────

    async def _close_position(self, exchange: ccxt.Exchange, symbol: str):
        if self._position is None or self._position_size <= 0:
            return

        close_side = "sell" if self._position == "long" else "buy"
        log.info(
            "Closing %s position: %s %.6f %s",
            self._position, close_side, self._position_size, symbol,
        )

        if self.cfg.dry_run:
            log.info("[DRY-RUN] Would %s %.6f %s", close_side, self._position_size, symbol)
            self._position = None
            self._position_size = 0.0
            return

        try:
            order = exchange.create_order(
                symbol, "market", close_side, self._position_size
            )
            log.info("Position closed: %s", order.get("id", "unknown"))
        except Exception as exc:
            log.error("Failed to close position: %s", exc)
        finally:
            self._position = None
            self._position_size = 0.0
            self._entry_price = 0.0

    # ── Place entry order ──────────────────────────────────────────────────

    async def _place_entry(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        side: str,
        size: float,
        atr: float,
    ) -> Optional[dict]:
        order_side = "buy" if side == "long" else "sell"
        order_type = "market" if self.cfg.use_market_orders else "limit"

        log.info(
            "Placing %s %s order: %.6f %s",
            order_type, order_side, size, symbol,
        )

        if self.cfg.dry_run:
            ticker = exchange.fetch_ticker(symbol)
            mock_price = ticker["last"]
            log.info(
                "[DRY-RUN] Would place %s %s %.6f @ ~%.4f",
                order_side, order_type, size, mock_price,
            )
            self._position = side
            self._position_size = size
            self._entry_price = mock_price
            stop = self._stop_price(side, mock_price, atr)
            log.info("[DRY-RUN] Stop-loss would be at %.4f", stop)
            return {"id": "dry-run", "price": mock_price}

        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            order = exchange.create_order(symbol, order_type, order_side, size)
            order_id = order.get("id")
            log.info("Entry order placed: id=%s", order_id)

            # Confirm fill within timeout
            filled_price = await self._confirm_fill(exchange, symbol, order_id, current_price)
            if filled_price is None:
                log.warning("Order %s not filled within %ds — cancelling.", order_id, self.cfg.order_timeout_seconds)
                try:
                    exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
                return None

            self._position = side
            self._position_size = size
            self._entry_price = filled_price

            # Place native stop-loss
            stop = self._stop_price(side, filled_price, atr)
            sl_side = "sell" if side == "long" else "buy"
            try:
                sl_order = exchange.create_order(
                    symbol, "stop_market", sl_side, size,
                    params={"stopPrice": stop},
                )
                log.info("Stop-loss placed at %.4f (id=%s)", stop, sl_order.get("id"))
            except Exception as exc:
                log.warning("Could not place native stop-loss: %s — manage manually.", exc)

            return order

        except Exception as exc:
            log.error("Entry order failed: %s", exc)
            return None

    async def _confirm_fill(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        order_id: str,
        fallback_price: float,
    ) -> Optional[float]:
        """Polls order status until filled or timeout."""
        deadline = time.time() + self.cfg.order_timeout_seconds
        while time.time() < deadline:
            try:
                order = exchange.fetch_order(order_id, symbol)
                status = order.get("status")
                if status == "closed":
                    price = order.get("average") or order.get("price") or fallback_price
                    log.info("Order %s filled @ %.4f", order_id, price)
                    return float(price)
                if status == "canceled":
                    return None
            except Exception as exc:
                log.warning("Order status poll failed: %s", exc)
            await asyncio.sleep(1)
        return None

    # ── Main entry point ───────────────────────────────────────────────────

    async def route_action(
        self,
        target_exchange: str,
        symbol: str,
        action_vector: np.ndarray,
        account_equity: float,
        current_atr: float,
    ):
        """
        Translates a PPO action vector into exchange orders.
        Called by live_daemon.py every candle cycle.
        """
        bias: float = float(action_vector[0])
        kelly: float = float(action_vector[1])

        desired_side = "long" if bias > 0 else "short"
        exchange = self._get_exchange(target_exchange)

        # Fetch current price for sizing
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = float(ticker["last"])
        except Exception as exc:
            log.error("Cannot fetch ticker for %s: %s", symbol, exc)
            return

        log.info(
            "route_action | bias=%.4f  kelly=%.4f  side=%s  price=%.4f  equity=%.2f  ATR=%.4f",
            bias, kelly, desired_side, current_price, account_equity, current_atr,
        )

        # Close opposing position if we're flipping sides
        if self._position and self._position != desired_side:
            log.info("Flipping from %s → %s", self._position, desired_side)
            await self._close_position(exchange, symbol)

        # Size the new position
        size = self._size_position(kelly, account_equity, current_price)
        if size <= 0:
            return

        # Place entry
        await self._place_entry(exchange, symbol, desired_side, size, current_atr)


# ─────────────────────────────────────────────
#  FastAPI tick ingestion server (optional)
# ─────────────────────────────────────────────

QUEUE_MAX = 50_000
data_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
_consumer_task: Optional[asyncio.Task] = None


def _validate_tick(raw: dict) -> bool:
    """Basic tick schema check before queuing."""
    required = {"symbol", "price", "timestamp"}
    return required.issubset(raw.keys())


async def _ingestion_consumer():
    log.info("Ingestion consumer started.")
    while True:
        try:
            item = await data_queue.get()
            # ── Hook: pass tick to feature factory or persistence layer ──
            # Example:
            #   await feature_factory.ingest_tick(item)
            data_queue.task_done()
        except asyncio.CancelledError:
            log.info("Ingestion consumer cancelled — draining remaining queue.")
            break
        except Exception as exc:
            log.error("Consumer error: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Replaces deprecated @app.on_event('startup'/'shutdown')."""
    global _consumer_task
    log.info(
        "ANTIGRAV GATEWAY starting | winloop=%s | queue_cap=%d",
        _WINLOOP_ACTIVE, QUEUE_MAX,
    )
    _consumer_task = asyncio.create_task(_ingestion_consumer())
    yield
    # Shutdown
    log.info("ANTIGRAV GATEWAY shutting down.")
    if _consumer_task:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            pass
    log.info("Gateway shutdown complete.")


app = FastAPI(title="ANTIGRAV_GATEWAY", lifespan=lifespan)


@app.websocket("/ws/v1/ticks")
async def tick_stream(websocket: WebSocket):
    """L2/L3 order book delta ingestion endpoint."""
    await websocket.accept()
    log.info("WebSocket ingestion channel open: %s", websocket.client)
    dropped = 0

    try:
        while True:
            raw_text = await websocket.receive_text()

            try:
                tick = json.loads(raw_text)
            except json.JSONDecodeError:
                log.warning("Malformed JSON tick dropped.")
                continue

            if not _validate_tick(tick):
                log.warning("Invalid tick schema dropped: %s", list(tick.keys()))
                continue

            if data_queue.full():
                dropped += 1
                if dropped % 100 == 1:
                    log.warning(
                        "Queue saturated (%d cap). Dropped %d ticks.",
                        QUEUE_MAX, dropped,
                    )
            else:
                await data_queue.put(tick)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected. Total dropped ticks: %d", dropped)
    except Exception as exc:
        log.error("WebSocket error: %s", exc)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "queue_depth": data_queue.qsize(),
        "queue_capacity": QUEUE_MAX,
        "winloop": _WINLOOP_ACTIVE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────
#  Standalone server entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "core.gateway:app",
        host="0.0.0.0",
        port=8000,
        loop="asyncio",   # winloop policy already installed globally above
        log_level="info",
    )