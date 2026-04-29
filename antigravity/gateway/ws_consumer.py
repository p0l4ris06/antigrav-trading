"""
WebSocket Queue Consumer.

Drains the bounded asyncio.Queue in configurable batches, routing ticks to:
    1. ClickHouse for persistence
    2. Feature Factory for real-time feature computation
    3. RL Agent for live inference (portfolio weights)
    4. Overseer for return tracking

Implements exponential backoff on ClickHouse write failures to prevent
queue backup cascades.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from antigravity.config import settings
from antigravity.tracing import get_tracer, span

if TYPE_CHECKING:
    from antigravity.db.client import ClickHouseManager
    from antigravity.features.factory import FeatureFactory
    from antigravity.overseer.daemon import AgenticOverseer
    from antigravity.regime.classifier import RegimeClassifier
    from antigravity.rl.agent import AgentManager

logger = structlog.get_logger(__name__)


class TickConsumer:
    """
    Async consumer that drains the tick queue and routes to persistence,
    feature computation, RL inference, and overseer monitoring.

    Design:
        - Batches ticks up to `batch_size` or `flush_interval_ms` (whichever
          comes first) to amortize ClickHouse insert overhead
        - Exponential backoff on CH failures (1s → 2s → 4s → ... → 30s cap)
        - Graceful degradation: if CH is unavailable, ticks still flow to
          the feature factory for real-time inference
    """

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        ch_manager: "ClickHouseManager | None",
        app_state: Any,
        feature_factory: "FeatureFactory | None" = None,
        regime_classifier: "RegimeClassifier | None" = None,
        agent: "AgentManager | None" = None,
        overseer: "AgenticOverseer | None" = None,
        batch_size: int | None = None,
        flush_interval_ms: int | None = None,
    ) -> None:
        self._queue = queue
        self._ch_manager = ch_manager
        self._app_state = app_state
        self._features = feature_factory
        self._regime = regime_classifier
        self._agent = agent
        self._overseer = overseer
        self._batch_size = batch_size or settings.gateway.batch_size
        self._flush_interval = (flush_interval_ms or settings.gateway.batch_flush_interval_ms) / 1000
        self._running = True
        self._backoff = 1.0  # exponential backoff for CH failures
        self._max_backoff = 30.0
        self._batch: list[dict[str, Any]] = []

        # Track last inference price for return calculation
        self._last_price: float | None = None

    def stop(self) -> None:
        """Signal the consumer to stop after draining current batch."""
        self._running = False

    async def run(self) -> None:
        """
        Main consumer loop.

        Collects ticks from the queue into a batch buffer. Flushes when:
            - Batch reaches `batch_size`, OR
            - `flush_interval_ms` elapses since last flush

        This dual trigger ensures both throughput (large batches during
        high volume) and latency (timely flush during low volume).
        """
        logger.info(
            "consumer.started",
            batch_size=self._batch_size,
            flush_interval_ms=self._flush_interval * 1000,
        )

        while self._running:
            try:
                # Try to fill the batch up to batch_size, with timeout
                try:
                    tick = await asyncio.wait_for(
                        self._queue.get(), timeout=self._flush_interval
                    )
                    self._prepare_tick(tick)
                    self._batch.append(tick)
                except asyncio.TimeoutError:
                    pass

                # Drain additional available items without blocking
                while len(self._batch) < self._batch_size:
                    try:
                        tick = self._queue.get_nowait()
                        self._prepare_tick(tick)
                        self._batch.append(tick)
                    except asyncio.QueueEmpty:
                        break

                # Flush if we have data
                if self._batch:
                    await self._flush_batch()

            except asyncio.CancelledError:
                # Final flush on shutdown
                if self._batch:
                    await self._flush_batch()
                raise
            except Exception as exc:
                logger.error("consumer.unexpected_error", error=str(exc))
                await asyncio.sleep(1.0)

        logger.info("consumer.stopped")

    def _prepare_tick(self, tick: dict[str, Any]) -> None:
        """Ensure tick has proper timestamp type for ClickHouse."""
        ts = tick.get("timestamp")
        if isinstance(ts, str):
            tick["timestamp"] = datetime.fromisoformat(ts)
        elif ts is None:
            tick["timestamp"] = datetime.now(timezone.utc)

    async def _flush_batch(self) -> None:
        """Flush the current batch to ClickHouse, feature factory, and inference pipeline."""
        batch = self._batch
        self._batch = []
        count = len(batch)

        tracer = get_tracer("antigravity.consumer")
        with tracer.start_as_current_span("flush_batch") as flush_span:
            flush_span.set_attribute("batch.size", count)
            flush_span.set_attribute("queue.depth", self._queue.qsize())

            # Route to ClickHouse (persistence)
            if self._ch_manager is not None:
                with tracer.start_as_current_span("clickhouse_insert") as ch_span:
                    ch_span.set_attribute("batch.size", count)
                    try:
                        await self._ch_manager.insert_tick_batch(batch)
                        self._backoff = 1.0  # reset on success
                        ch_span.set_attribute("status", "ok")
                    except Exception as exc:
                        ch_span.set_attribute("status", "error")
                        ch_span.set_attribute("error.message", str(exc))
                        logger.error(
                            "consumer.ch_insert_failed",
                            error=str(exc),
                            batch_size=count,
                            backoff=self._backoff,
                        )
                        await asyncio.sleep(self._backoff)
                        self._backoff = min(self._backoff * 2, self._max_backoff)

            # Route to Feature Factory + RL inference pipeline
            await self._route_to_features(batch)

        logger.debug("consumer.flushed", count=count, queue_depth=self._queue.qsize())

    async def _route_to_features(self, batch: list[dict[str, Any]]) -> None:
        """
        Full inference pipeline:
            1. Ingest ticks → Feature Factory
            2. Compute features → Regime Classifier
            3. Concatenate features + regime → RL Agent
            4. Agent produces portfolio weights → AppState
            5. Track returns → Overseer
        """
        if not batch:
            return

        tracer = get_tracer("antigravity.consumer")

        # --- Step 1: Ingest into Feature Factory ---
        if self._features is not None:
            with tracer.start_as_current_span("feature_ingest") as fi_span:
                self._features.ingest_batch(batch)
                fi_span.set_attribute("batch.size", len(batch))
                fi_span.set_attribute("buffer.height", self._features.buffer_height)

            # Update MFE/MAE tracking with latest price
            last_tick = batch[-1]
            last_price = float(last_tick.get("last_price", 0))

            if last_price > 0:
                if self._features._entry_price is None:
                    self._features.open_trade(last_price)
                else:
                    self._features.update_trade(last_price)

            # --- Step 2: Compute features and get observation ---
            with tracer.start_as_current_span("feature_compute") as fc_span:
                obs = self._features.get_latest_features()
                fc_span.set_attribute("obs.available", obs is not None)
                if obs is not None:
                    fc_span.set_attribute("obs.dim", len(obs))

            if obs is not None:
                # --- Step 3: Regime classification ---
                regime_proba = np.zeros(3, dtype=np.float32)
                if self._regime is not None and self._regime.is_fitted:
                    with tracer.start_as_current_span("regime_classify") as rc_span:
                        try:
                            proba = self._regime.predict_proba(obs)
                            if proba.ndim == 2:
                                regime_proba = proba[0].astype(np.float32)
                            else:
                                regime_proba = proba.astype(np.float32)

                            regime_id = int(np.argmax(regime_proba))
                            regime_name = self._regime.get_regime_name(regime_id)
                            self._app_state.current_regime = regime_name
                            self._app_state.regime_probabilities = regime_proba.tolist()

                            rc_span.set_attribute("regime.id", regime_id)
                            rc_span.set_attribute("regime.name", regime_name)
                            rc_span.set_attribute("regime.confidence", float(regime_proba.max()))
                        except Exception as exc:
                            rc_span.set_attribute("status", "error")
                            logger.debug("consumer.regime_error", error=str(exc))

                # --- Step 4: RL Agent inference ---
                if self._agent is not None:
                    with tracer.start_as_current_span("rl_predict") as rl_span:
                        try:
                            # Concatenate features + regime probabilities
                            full_obs = np.concatenate([obs, regime_proba]).astype(np.float32)
                            weights = self._agent.predict(full_obs, deterministic=True)
                            self._app_state.portfolio_weights = weights.tolist()

                            rl_span.set_attribute("obs.dim", len(full_obs))
                            rl_span.set_attribute("weights", str(weights.tolist()))

                            # --- Step 5: Track returns for Overseer ---
                            if last_price > 0 and self._last_price is not None and self._last_price > 0:
                                price_return = (last_price - self._last_price) / self._last_price
                                portfolio_return = float(np.sum(weights * price_return))
                                rl_span.set_attribute("portfolio.return", portfolio_return)

                                if self._overseer is not None:
                                    self._overseer.add_return(portfolio_return)

                        except Exception as exc:
                            rl_span.set_attribute("status", "error")
                            logger.debug("consumer.agent_error", error=str(exc))

                if last_price > 0:
                    self._last_price = last_price
                    # Update global price tracker
                    symbol = last_tick.get("symbol", "UNKNOWN")
                    self._app_state.prices[symbol] = last_price

