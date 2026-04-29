"""
Agentic Overseer — Self-Healing Autonomous Daemon.

Implements a deterministic state machine that:
    1. Monitors out-of-sample (OOS) rolling Sharpe ratio
    2. Detects concept drift via Page-Hinkley sequential test
    3. Spawns shadow fork training in a separate OS process
    4. Validates new policy via paired t-test
    5. Hot-swaps model weights if statistically significant improvement

State Machine:
    MONITORING → DRIFT_DETECTED → SHADOW_TRAINING → VALIDATION
    VALIDATION → HOT_SWAP (p < 0.05) | MONITORING (p ≥ 0.05)
    HOT_SWAP → MONITORING
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
import threading
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from scipy import stats

from antigravity.config import settings
from antigravity.tracing import get_tracer

if TYPE_CHECKING:
    from antigravity.db.client import ClickHouseManager
    from antigravity.features.factory import FeatureFactory
    from antigravity.regime.classifier import RegimeClassifier
    from antigravity.rl.agent import AgentManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Page-Hinkley Drift Detector
# ---------------------------------------------------------------------------
class PageHinkley:
    """
    Page-Hinkley sequential change-point detection.

    Monitors a stream of values and detects distributional shift when the
    cumulative deviation from the running mean exceeds a threshold.

    Parameters:
        delta:     minimum magnitude of allowed change (sensitivity)
        threshold: detection threshold (higher = fewer false alarms)
        alpha:     forgetting factor for the running mean (unused in classic PH)
    """

    def __init__(
        self,
        delta: float | None = None,
        threshold: float | None = None,
    ) -> None:
        cfg = settings.overseer
        self.delta = delta or cfg.drift_delta
        self.threshold = threshold or cfg.drift_threshold

        self._n: int = 0
        self._sum: float = 0.0
        self._mean: float = 0.0
        self._min_sum: float = float("inf")

    def update(self, value: float) -> bool:
        """
        Feed a new observation. Returns True if drift is detected.

        The test statistic is:
            PH_t = Σ_{i=1}^{t} (x_i - x̄_t - δ)
            min_PH = min(PH_1, ..., PH_t)
            Drift detected when PH_t - min_PH > threshold
        """
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        drift = (self._sum - self._min_sum) > self.threshold

        if drift:
            logger.warning(
                "drift.detected",
                statistic=round(self._sum - self._min_sum, 4),
                threshold=self.threshold,
                n_observations=self._n,
            )

        return drift

    def reset(self) -> None:
        """Reset the detector state after a drift event is handled."""
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._min_sum = float("inf")


# ---------------------------------------------------------------------------
# Overseer State Machine
# ---------------------------------------------------------------------------
class OverseerState(Enum):
    MONITORING = auto()
    DRIFT_DETECTED = auto()
    SHADOW_TRAINING = auto()
    VALIDATION = auto()
    HOT_SWAP = auto()


class AgenticOverseer:
    """
    Autonomous daemon that makes the system self-healing.

    Runs as an asyncio task within the gateway process. Uses a
    deterministic state machine to manage drift detection, shadow
    fork training, validation, and weight hot-swap.
    """

    def __init__(
        self,
        agent: "AgentManager | None" = None,
        feature_factory: "FeatureFactory | None" = None,
        regime_classifier: "RegimeClassifier | None" = None,
        ch_manager: "ClickHouseManager | None" = None,
        app_state: Any = None,
    ) -> None:
        self._agent = agent
        self._features = feature_factory
        self._regime = regime_classifier
        self._ch_manager = ch_manager
        self._app_state = app_state

        self._state = OverseerState.MONITORING
        self._drift_detector = PageHinkley()
        self._sharpe_window = settings.overseer.sharpe_window
        self._swap_p_value = settings.overseer.swap_p_value
        self._shadow_timesteps = settings.overseer.shadow_train_timesteps

        # Return tracking for rolling Sharpe
        self._returns: deque[float] = deque(maxlen=self._sharpe_window)

        # Shadow fork process management
        self._shadow_process: mp.Process | None = None
        self._result_queue: mp.Queue = mp.Queue()
        self._swap_lock = threading.Lock()

        # Event log for dashboard
        self._event_log: deque[dict[str, Any]] = deque(maxlen=200)

    @property
    def state(self) -> str:
        return self._state.name

    @property
    def event_log(self) -> list[dict[str, Any]]:
        return list(self._event_log)

    # ------------------------------------------------------------------
    # Rolling Sharpe Calculation
    # ------------------------------------------------------------------

    def _rolling_sharpe(self) -> float:
        """
        Calculate the rolling Sharpe ratio over the configured window.
        48-hour at 1-min bars = 2880 observations.
        """
        if len(self._returns) < 100:
            return 0.0
        r = np.array(self._returns)
        std = np.std(r)
        if std < 1e-10:
            return 0.0
        return float(np.mean(r) / std * np.sqrt(len(r)))

    # ------------------------------------------------------------------
    # State Machine Execution
    # ------------------------------------------------------------------

    async def run(self, poll_interval: float = 5.0) -> None:
        """
        Main overseer loop. Checks system health at regular intervals.

        State transitions:
            MONITORING      → check Sharpe, feed drift detector
            DRIFT_DETECTED  → spawn shadow fork process
            SHADOW_TRAINING → poll result queue
            VALIDATION      → paired t-test on shadow vs live returns
            HOT_SWAP        → swap weights, reset detectors
        """
        logger.info("overseer.started", poll_interval=poll_interval)
        self._log_event("system_monitoring_started", status="OPTIMAL", poll_interval=poll_interval)

        while True:
            try:
                tracer = get_tracer("antigravity.overseer")
                with tracer.start_as_current_span("overseer_tick") as tick_span:
                    tick_span.set_attribute("state", self._state.name)
                    tick_span.set_attribute("rolling_sharpe", self._rolling_sharpe())
                    tick_span.set_attribute("n_returns", len(self._returns))

                    prev_state = self._state

                    match self._state:
                        case OverseerState.MONITORING:
                            await self._handle_monitoring()

                        case OverseerState.DRIFT_DETECTED:
                            await self._handle_drift_detected()

                        case OverseerState.SHADOW_TRAINING:
                            await self._handle_shadow_training()

                        case OverseerState.VALIDATION:
                            await self._handle_validation()

                        case OverseerState.HOT_SWAP:
                            await self._handle_hot_swap()

                    # Record state transitions
                    if self._state != prev_state:
                        tick_span.set_attribute("transition.from", prev_state.name)
                        tick_span.set_attribute("transition.to", self._state.name)

                # Update app state for dashboard
                if self._app_state:
                    self._app_state.overseer_state = self._state.name
                    self._app_state.rolling_sharpe = self._rolling_sharpe()
                    self._app_state.drift_detected = (
                        self._state == OverseerState.DRIFT_DETECTED
                    )
                    self._app_state.shadow_fork_active = (
                        self._state == OverseerState.SHADOW_TRAINING
                    )

                # Check for force-retrain signals or proactive mode
                if (
                    (self._app_state and getattr(self._app_state, "overseer_state", "") == "FORCE_RETRAIN_REQUESTED")
                    or (settings.overseer.proactive_shadow and self._state == OverseerState.MONITORING)
                ):
                    is_manual = (self._app_state and getattr(self._app_state, "overseer_state", "") == "FORCE_RETRAIN_REQUESTED")
                    self._log_event("shadow_fork_triggered", reason="manual" if is_manual else "proactive")
                    self._state = OverseerState.DRIFT_DETECTED
                    
                    # Clear manual request to prevent infinite loop
                    if is_manual and self._app_state:
                        self._app_state.overseer_state = self._state.name

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                logger.info("overseer.cancelled")
                self._cleanup_shadow()
                raise
            except Exception as exc:
                logger.error("overseer.error", error=str(exc))
                await asyncio.sleep(poll_interval * 2)

    # ------------------------------------------------------------------
    # State Handlers
    # ------------------------------------------------------------------

    async def _handle_monitoring(self) -> None:
        """MONITORING: calculate Sharpe, feed Page-Hinkley."""
        sharpe = self._rolling_sharpe()

        # Feed the drift detector with negative Sharpe (detect degradation)
        if len(self._returns) >= 100:
            drift = self._drift_detector.update(-sharpe)
            if drift:
                self._log_event(
                    "drift_detected",
                    sharpe=round(sharpe, 4),
                )
                self._state = OverseerState.DRIFT_DETECTED

        # Check regime classifier refit
        if self._regime and self._features:
            latest = self._features.get_latest_features()
            if latest is not None and self._regime.is_fitted:
                if self._regime.should_refit(latest):
                    self._log_event("regime_refit_triggered")
                    # Get recent feature data for refitting
                    df = self._features.compute_features()
                    if df is not None:
                        feature_cols = self._features.get_feature_names()
                        available = [c for c in feature_cols if c in df.columns]
                        if available:
                            matrix = df.select(available).drop_nulls().to_numpy()
                            self._regime.fit(matrix)
                            self._log_event("regime_refitted")

    async def _handle_drift_detected(self) -> None:
        """DRIFT_DETECTED: spawn shadow fork training process."""
        if self._agent is None:
            logger.warning("overseer.no_agent_configured")
            self._state = OverseerState.MONITORING
            return

        self._log_event("shadow_fork_spawning")

        # Clone current weights
        current_weights = self._agent.clone_weights()

        # Export recent data to Parquet for the shadow fork
        data_path = None
        if self._ch_manager:
            try:
                now = datetime.now(timezone.utc)
                start = now - timedelta(hours=48)
                data_path = await self._ch_manager.export_parquet(
                    symbol="BTCUSDT",
                    start=start,
                    end=now,
                    path=Path(tempfile.gettempdir()) / "antigravity_shadow_data.parquet",
                )
            except Exception as exc:
                logger.error("overseer.data_export_failed", error=str(exc))

        # Spawn shadow training in separate process
        self._shadow_process = mp.Process(
            target=_shadow_train_worker,
            args=(
                current_weights,
                str(data_path) if data_path else None,
                self._result_queue,
                self._shadow_timesteps,
            ),
            daemon=True,
        )
        self._shadow_process.start()
        self._log_event("shadow_fork_started", pid=self._shadow_process.pid)
        self._state = OverseerState.SHADOW_TRAINING

    async def _handle_shadow_training(self) -> None:
        """SHADOW_TRAINING: poll result queue for completion."""
        if self._shadow_process is None:
            self._state = OverseerState.MONITORING
            return

        # Non-blocking check
        if not self._shadow_process.is_alive():
            try:
                result = self._result_queue.get_nowait()
                self._shadow_result = result
                self._log_event(
                    "shadow_fork_complete",
                    success=result.get("success", False),
                )
                self._state = OverseerState.VALIDATION
            except Exception:
                self._log_event("shadow_fork_failed")
                self._state = OverseerState.MONITORING
                self._cleanup_shadow()
        elif not self._result_queue.empty():
            result = self._result_queue.get_nowait()
            self._shadow_result = result
            self._state = OverseerState.VALIDATION

    async def _handle_validation(self) -> None:
        """VALIDATION: paired t-test on shadow vs current returns."""
        result = getattr(self, "_shadow_result", None)
        if result is None or not result.get("success", False):
            self._state = OverseerState.MONITORING
            self._cleanup_shadow()
            return

        new_returns = np.array(result.get("new_returns", []))
        old_returns = np.array(result.get("old_returns", []))

        if len(new_returns) < 10 or len(old_returns) < 10:
            self._log_event("validation_insufficient_data")
            self._state = OverseerState.MONITORING
            self._cleanup_shadow()
            return

        # Paired t-test: is the new policy statistically better?
        min_len = min(len(new_returns), len(old_returns))
        t_stat, p_value = stats.ttest_rel(
            new_returns[:min_len], old_returns[:min_len]
        )

        new_better = np.mean(new_returns) > np.mean(old_returns)
        significant = p_value < self._swap_p_value

        self._log_event(
            "validation_complete",
            p_value=round(float(p_value), 6),
            t_statistic=round(float(t_stat), 4),
            new_mean=round(float(np.mean(new_returns)), 6),
            old_mean=round(float(np.mean(old_returns)), 6),
            significant=bool(significant),
            new_better=bool(new_better),
        )

        if significant and new_better:
            self._new_weights = result.get("new_weights")
            self._state = OverseerState.HOT_SWAP
        else:
            logger.info(
                "overseer.shadow_rejected",
                p_value=round(float(p_value), 6),
            )
            self._state = OverseerState.MONITORING
            self._cleanup_shadow()

    async def _handle_hot_swap(self) -> None:
        """HOT_SWAP: swap weights and reset detectors."""
        new_weights = getattr(self, "_new_weights", None)
        if new_weights is not None and self._agent is not None:
            self._agent.load_weights(new_weights)
            self._log_event("weights_hot_swapped")
            logger.info("overseer.hot_swap_complete")

        # Reset drift detector
        self._drift_detector.reset()
        self._cleanup_shadow()
        self._state = OverseerState.MONITORING

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def add_return(self, portfolio_return: float) -> None:
        """Feed a new portfolio return observation to the overseer."""
        self._returns.append(portfolio_return)

    def _cleanup_shadow(self) -> None:
        """Clean up shadow fork process resources."""
        if self._shadow_process and self._shadow_process.is_alive():
            self._shadow_process.terminate()
            self._shadow_process.join(timeout=5)
        self._shadow_process = None
        self._shadow_result = None
        self._new_weights = None

    def _log_event(self, event: str, **kwargs: Any) -> None:
        """Log an overseer event for the dashboard."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        self._event_log.append(entry)
        logger.info(f"overseer.{event}", **kwargs)


# ---------------------------------------------------------------------------
# Shadow Fork Worker (runs in separate OS process)
# ---------------------------------------------------------------------------
def _shadow_train_worker(
    current_weights: dict[str, Any],
    data_path: str | None,
    result_queue: mp.Queue,
    timesteps: int,
) -> None:
    """
    Shadow fork training process.

    Runs in a completely separate OS process (via multiprocessing):
        1. Loads recent data from Parquet (or generates synthetic)
        2. Clones current policy weights
        3. Trains new policy on recent data
        4. Backtests both policies on holdout period
        5. Puts results on the queue for the overseer to validate

    This function must be self-contained — it cannot access the
    parent process's memory or asyncio loop.
    """
    try:
        import polars as pl
        from antigravity.rl.environment import TradingEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # Load data
        if data_path and Path(data_path).exists():
            df = pl.read_parquet(data_path)
            prices = df.select("last_price").to_numpy().flatten()
            features_raw = df.select(
                [c for c in df.columns if c not in
                 {"symbol", "timestamp", "trade_id"}]
            ).to_numpy()
        else:
            # Synthetic fallback for development
            n = 5000
            prices = 50000 + np.cumsum(np.random.randn(n) * 10)
            features_raw = np.random.randn(n, 10)

        # Compute basic ATR proxy
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        from scipy.ndimage import uniform_filter1d
        atr = uniform_filter1d(price_changes, size=14)

        # Split: 80% train, 20% holdout
        split = int(len(prices) * 0.8)
        train_features = features_raw[:split].astype(np.float32)
        train_prices = prices[:split]
        train_atr = atr[:split]

        holdout_features = features_raw[split:].astype(np.float32)
        holdout_prices = prices[split:]
        holdout_atr = atr[split:]

        # Create training environment
        train_env = TradingEnv(
            feature_data=train_features,
            price_data=train_prices,
            atr_data=train_atr,
            episode_length=min(400, split - 1),
        )

        vec_env = DummyVecEnv([lambda: train_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        # Initialize PPO with cloned weights
        model = PPO("MlpPolicy", vec_env, verbose=0, device="auto")
        try:
            model.policy.load_state_dict(current_weights)
        except Exception:
            pass  # Architecture mismatch — train from scratch

        # Train
        model.learn(total_timesteps=timesteps)

        # Backtest on holdout
        holdout_env = TradingEnv(
            feature_data=holdout_features,
            price_data=holdout_prices,
            atr_data=holdout_atr,
            episode_length=min(400, len(holdout_prices) - 1),
        )

        new_returns = _backtest_policy(model, holdout_env)

        # Backtest old policy (clone weights back)
        old_model = PPO("MlpPolicy", vec_env, verbose=0, device="auto")
        try:
            old_model.policy.load_state_dict(current_weights)
        except Exception:
            pass

        old_returns = _backtest_policy(old_model, holdout_env)

        result_queue.put({
            "success": True,
            "new_weights": model.policy.state_dict(),
            "new_returns": new_returns,
            "old_returns": old_returns,
        })

    except Exception as exc:
        result_queue.put({
            "success": False,
            "error": str(exc),
            "new_returns": [],
            "old_returns": [],
        })


def _backtest_policy(model: Any, env: "TradingEnv", n_episodes: int = 5) -> list[float]:
    """Run a policy through the environment and collect per-step returns."""
    all_returns: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            all_returns.append(info.get("portfolio_return", 0.0))
            done = terminated or truncated

    return all_returns
