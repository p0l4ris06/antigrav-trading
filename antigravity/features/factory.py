"""
Feature Factory — Zero-Copy Vectorized Feature Engineering.

All computation uses Polars expressions (Rust-backed, SIMD-accelerated,
bypasses the GIL). No Python-level iteration over tick data.

Core Features:
    - ATR (Average True Range) via EMA of True Range
    - OBI (Order Book Imbalance): (V_bid - V_ask) / (V_bid + V_ask)
    - Microprice: (bid × ask_size + ask × bid_size) / (bid_size + ask_size)
    - VWAP over multiple temporal windows
    - Log returns and rolling volatility
    - MFE / MAE tracking per trade lifecycle
    - Spread and volume ratio

Agentic Interface:
    - prune_correlated_features() drops features exceeding Spearman
      rank correlation threshold to prevent curse of dimensionality
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import polars as pl
import structlog
from scipy.stats import spearmanr

from antigravity.config import settings

logger = structlog.get_logger(__name__)


class FeatureFactory:
    """
    Stateful feature engineering pipeline operating on a rolling tick buffer.

    Design:
        - Maintains a fixed-size ring buffer of raw ticks (Polars DataFrame)
        - All feature computation is lazy-evaluated Polars expressions
        - No Python GC pressure: zero-copy Arrow-backed memory
        - Exposes an agentic interface for autonomous feature selection
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        atr_period: int | None = None,
        vwap_windows: list[int] | None = None,
        vol_windows: list[int] | None = None,
    ) -> None:
        cfg = settings.features
        self._buffer_size = buffer_size or cfg.tick_buffer_size
        self._atr_period = atr_period or cfg.atr_period
        self._vwap_windows = vwap_windows or cfg.vwap_windows
        self._vol_windows = vol_windows or cfg.rolling_vol_windows

        # Raw tick buffer
        self._buffer: pl.DataFrame = pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "timestamp": pl.Datetime("us", "UTC"),
                "bid_price": pl.Float64,
                "ask_price": pl.Float64,
                "bid_size": pl.Float64,
                "ask_size": pl.Float64,
                "last_price": pl.Float64,
                "last_size": pl.Float64,
            }
        )

        # Trade lifecycle tracking for MFE/MAE
        self._entry_price: float | None = None
        self._mfe: float = 0.0
        self._mae: float = 0.0

        # Feature name registry (for agentic pruning)
        self._active_features: list[str] = []
        self._dropped_features: set[str] = set()

    @property
    def buffer_height(self) -> int:
        """Current number of ticks in buffer."""
        return self._buffer.height

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_tick(self, tick: dict[str, Any]) -> None:
        """Add a single tick to the buffer."""
        row = pl.DataFrame(
            {
                "symbol": [tick.get("symbol", "UNKNOWN")],
                "timestamp": [tick.get("timestamp")],
                "bid_price": [float(tick.get("bid_price", 0))],
                "ask_price": [float(tick.get("ask_price", 0))],
                "bid_size": [float(tick.get("bid_size", 0))],
                "ask_size": [float(tick.get("ask_size", 0))],
                "last_price": [float(tick.get("last_price", 0))],
                "last_size": [float(tick.get("last_size", 0))],
            },
            schema=self._buffer.schema,
        )
        self._buffer = pl.concat([self._buffer, row])
        if self._buffer.height > self._buffer_size:
            self._buffer = self._buffer.tail(self._buffer_size)

    def ingest_batch(self, ticks: list[dict[str, Any]]) -> None:
        """Add a batch of ticks to the buffer."""
        if not ticks:
            return
        batch = pl.DataFrame(
            {
                "symbol": [t.get("symbol", "UNKNOWN") for t in ticks],
                "timestamp": [t.get("timestamp") for t in ticks],
                "bid_price": [float(t.get("bid_price", 0)) for t in ticks],
                "ask_price": [float(t.get("ask_price", 0)) for t in ticks],
                "bid_size": [float(t.get("bid_size", 0)) for t in ticks],
                "ask_size": [float(t.get("ask_size", 0)) for t in ticks],
                "last_price": [float(t.get("last_price", 0)) for t in ticks],
                "last_size": [float(t.get("last_size", 0)) for t in ticks],
            },
            schema=self._buffer.schema,
        )
        self._buffer = pl.concat([self._buffer, batch])
        if self._buffer.height > self._buffer_size:
            self._buffer = self._buffer.tail(self._buffer_size)

    # ------------------------------------------------------------------
    # Feature Computation (Pure Polars Expressions)
    # ------------------------------------------------------------------

    def compute_features(self, n_bars: int | None = None) -> pl.DataFrame | None:
        """
        Compute the full feature matrix from the tick buffer.

        Returns None if insufficient data (< 2 * max rolling window).
        All computations are vectorized Polars expressions — no Python loops.
        """
        min_required = max(self._atr_period, max(self._vol_windows)) * 2
        if self._buffer.height < min_required:
            return None

        df = self._buffer if n_bars is None else self._buffer.tail(n_bars)

        # --- Core price features ---
        df = df.with_columns(
            [
                # Log returns
                pl.col("last_price").log().diff().alias("log_returns"),
                # Spread
                (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
                # Order Book Imbalance: (V_bid - V_ask) / (V_bid + V_ask)
                (
                    (pl.col("bid_size") - pl.col("ask_size"))
                    / (pl.col("bid_size") + pl.col("ask_size") + 1e-10)
                ).alias("obi"),
                # Microprice: (bid × ask_size + ask × bid_size) / (bid_size + ask_size)
                (
                    (pl.col("bid_price") * pl.col("ask_size")
                     + pl.col("ask_price") * pl.col("bid_size"))
                    / (pl.col("bid_size") + pl.col("ask_size") + 1e-10)
                ).alias("microprice"),
                # Mid price
                ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"),
            ]
        )

        # --- True Range & ATR ---
        df = df.with_columns(
            [
                # True Range components (using mid price as proxy when no explicit H/L/C)
                # TR = max(H-L, |H-prevC|, |L-prevC|) ≈ spread + |mid_change| for tick data
                (
                    pl.col("spread")
                    + pl.col("mid_price").diff().abs().fill_null(0)
                ).alias("true_range"),
            ]
        )

        # ATR via EWM (exponential weighted moving average)
        df = df.with_columns(
            [
                pl.col("true_range")
                .ewm_mean(span=self._atr_period, adjust=False)
                .alias("atr"),
            ]
        )

        # --- Rolling volatility at multiple windows ---
        vol_exprs = []
        for w in self._vol_windows:
            vol_exprs.append(
                pl.col("log_returns").rolling_std(window_size=w).alias(f"vol_{w}")
            )
        df = df.with_columns(vol_exprs)

        # --- VWAP at multiple temporal windows ---
        # For tick data, VWAP = cumulative(price * volume) / cumulative(volume)
        # We approximate with rolling sums
        vwap_exprs = []
        for w in self._vwap_windows:
            # Convert seconds to approximate tick count (assume ~10 ticks/sec for sim)
            tick_window = min(w * 10, self._buffer.height)
            if tick_window < 2:
                tick_window = 2
            pv_col = f"_pv_{w}"
            v_col = f"_v_{w}"
            vwap_col = f"vwap_{w}s"

            vwap_exprs.extend(
                [
                    (
                        (pl.col("last_price") * pl.col("last_size"))
                        .rolling_sum(window_size=tick_window)
                    ).alias(pv_col),
                    pl.col("last_size")
                    .rolling_sum(window_size=tick_window)
                    .alias(v_col),
                ]
            )

        if vwap_exprs:
            df = df.with_columns(vwap_exprs)
            # Compute VWAP ratios
            for w in self._vwap_windows:
                tick_window = min(w * 10, self._buffer.height)
                if tick_window < 2:
                    continue
                pv_col = f"_pv_{w}"
                v_col = f"_v_{w}"
                vwap_col = f"vwap_{w}s"
                df = df.with_columns(
                    (pl.col(pv_col) / (pl.col(v_col) + 1e-10)).alias(vwap_col)
                )
                df = df.drop([pv_col, v_col])

        # --- Volume ratio ---
        df = df.with_columns(
            [
                (
                    pl.col("last_size")
                    / (pl.col("last_size").rolling_mean(window_size=50) + 1e-10)
                ).alias("volume_ratio"),
            ]
        )

        # --- Normalize features for RL consumption ---
        # ATR-normalized spread and microprice deviation
        df = df.with_columns(
            [
                (pl.col("spread") / (pl.col("atr") + 1e-10)).alias("spread_atr_norm"),
                (
                    (pl.col("microprice") - pl.col("mid_price"))
                    / (pl.col("atr") + 1e-10)
                ).alias("microprice_deviation"),
            ]
        )

        # Drop intermediate columns
        df = df.drop(["true_range", "mid_price"])

        # Register active features
        self._active_features = [
            c for c in df.columns
            if c not in {"symbol", "timestamp", "bid_price", "ask_price",
                         "bid_size", "ask_size", "last_price", "last_size"}
            and c not in self._dropped_features
        ]

        return df

    def get_latest_features(self) -> np.ndarray | None:
        """
        Compute features and return the latest row as a numpy vector.
        This is the primary interface for the RL agent's observation.
        """
        df = self.compute_features()
        if df is None:
            return None

        feature_cols = [c for c in self._active_features if c in df.columns]
        if not feature_cols:
            return None

        row = df.select(feature_cols).tail(1).to_numpy()
        # Replace NaN/inf with 0 for numerical stability
        return np.nan_to_num(row.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # MFE / MAE Trade Lifecycle Tracking
    # ------------------------------------------------------------------

    def open_trade(self, entry_price: float) -> None:
        """Mark the start of a new trade for MFE/MAE calculation."""
        self._entry_price = entry_price
        self._mfe = 0.0
        self._mae = 0.0

    def update_trade(self, current_price: float) -> tuple[float, float]:
        """
        Update MFE/MAE with current price.

        Returns:
            (mfe, mae) — peak unrealized profit and peak unrealized loss
        """
        if self._entry_price is None:
            return 0.0, 0.0

        pnl = current_price - self._entry_price
        self._mfe = max(self._mfe, pnl)
        self._mae = max(self._mae, -pnl)  # MAE is magnitude of adverse excursion
        return self._mfe, self._mae

    def close_trade(self) -> tuple[float, float]:
        """
        Close current trade and return final MFE/MAE.

        Returns:
            (mfe, mae) — final values for reward computation
        """
        mfe, mae = self._mfe, self._mae
        self._entry_price = None
        self._mfe = 0.0
        self._mae = 0.0
        return mfe, mae

    def get_current_atr(self) -> float:
        """Return the latest ATR value for reward normalization."""
        df = self.compute_features()
        if df is None or "atr" not in df.columns:
            return 1.0  # safe default to avoid division by zero
        atr_val = df.select("atr").tail(1).item()
        return float(atr_val) if atr_val is not None and atr_val > 0 else 1.0

    # ------------------------------------------------------------------
    # Agentic Feature Selection Interface
    # ------------------------------------------------------------------

    def prune_correlated_features(
        self,
        df: pl.DataFrame | None = None,
        threshold: float | None = None,
    ) -> list[str]:
        """
        Compute Spearman rank correlation matrix and iteratively drop
        features exceeding the threshold, keeping the feature with
        higher variance in each correlated pair.

        Args:
            df: Feature DataFrame (uses buffer if None)
            threshold: Spearman |ρ| above which to drop (default from config)

        Returns:
            List of dropped feature names
        """
        threshold = threshold or settings.features.correlation_threshold

        if df is None:
            df = self.compute_features()
        if df is None:
            return []

        feature_cols = [
            c for c in self._active_features
            if c in df.columns and c not in self._dropped_features
        ]
        if len(feature_cols) < 2:
            return []

        # Extract numeric matrix
        matrix = df.select(feature_cols).drop_nulls().to_numpy()
        if matrix.shape[0] < 10:
            return []

        # Compute Spearman correlation
        corr_matrix, _ = spearmanr(matrix)
        if corr_matrix.ndim == 0:
            return []

        # Make it a proper 2D array if only 2 features
        if corr_matrix.ndim == 1:
            corr_matrix = np.array([[1.0, corr_matrix], [corr_matrix, 1.0]])

        # Iterative pruning: drop lower-variance feature from each pair
        variances = np.var(matrix, axis=0)
        dropped: list[str] = []
        active_mask = np.ones(len(feature_cols), dtype=bool)

        for i in range(len(feature_cols)):
            if not active_mask[i]:
                continue
            for j in range(i + 1, len(feature_cols)):
                if not active_mask[j]:
                    continue
                if abs(corr_matrix[i, j]) > threshold:
                    # Drop the feature with lower variance
                    drop_idx = j if variances[i] >= variances[j] else i
                    active_mask[drop_idx] = False
                    dropped.append(feature_cols[drop_idx])

        # Update internal state
        self._dropped_features.update(dropped)
        self._active_features = [f for f in self._active_features if f not in self._dropped_features]

        if dropped:
            logger.info(
                "features.pruned_correlated",
                dropped=dropped,
                remaining=len(self._active_features),
                threshold=threshold,
            )

        return dropped

    def disable_feature(self, name: str) -> None:
        """Manually disable a specific feature."""
        self._dropped_features.add(name)
        if name in self._active_features:
            self._active_features.remove(name)

    def reset_feature_selection(self) -> None:
        """Re-enable all features (undo pruning)."""
        self._dropped_features.clear()
        logger.info("features.selection_reset")

    def get_feature_names(self) -> list[str]:
        """Return list of currently active feature names."""
        return list(self._active_features)
