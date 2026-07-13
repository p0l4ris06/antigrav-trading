# core/features_extended.py
# ─────────────────────────────────────────────────────────────────────────────
# ExtendedFeatureFactory — Phase 1 togglable feature module.
#
# Activates with:  --feature-set extended  in train.py / auto_optimizer.py
#
# What it adds on top of SMCFeatureFactory (base set):
#   ── 4H timeframe columns (computed on whatever OHLCV is supplied; designed
#      for 4H parquets but works on any timeframe):
#       trend_strength   — ADX-proxy (normalised ATR vs longer rolling ATR)
#       momentum_slope   — Linear regression slope of close over 10 bars (normalised)
#       vol_regime       — Volatility regime: rolling 20-bar vol / 100-bar vol
#       candle_body_ratio— Body / total range; filters doji chop
#
#   ── Microstructure / OBI columns (only added when present in the DataFrame):
#       obi              — Order Book Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
#       volume_delta     — Signed volume delta: (up_vol - dn_vol) / total_vol
#       funding_rate     — Raw funding rate if available (else 0.0)
#
# TOGGLE:
#   Pass `use_extended=True` to ExtendedFeatureFactory or set the env var
#   ANTIGRAV_EXTENDED_FEATURES=1 to enable.  Falls back cleanly to base-only
#   behaviour if env var is absent.
#
# CONSTRAINTS HONOURED:
#   - 100% Polars, zero Pandas
#   - All rolling ops are trailing (no center=True, no shift(-n))
#   - fill_nan(0.0).fill_null(0.0) on every derived column
#   - OBI/microstructure columns silently skipped if not in DataFrame schema
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import polars as pl

from antigravity.features.base import SMCFeatureFactory


# ── Feature flag ──────────────────────────────────────────────────────────────

_EXTENDED_ENV_VAR = "ANTIGRAV_EXTENDED_FEATURES"


def _use_extended_default() -> bool:
    return os.getenv(_EXTENDED_ENV_VAR, "0") == "1"


# ── Extended factory ──────────────────────────────────────────────────────────

class ExtendedFeatureFactory:
    """
    Wraps SMCFeatureFactory and optionally appends Phase 1 features.

    Parameters
    ----------
    swing_length : int
        Passed through to the base SMCFeatureFactory.
    use_extended : bool
        If True, compute and append the extended 4H + OBI feature columns.
        Defaults to the ANTIGRAV_EXTENDED_FEATURES environment variable.

    Usage in train.py::

        if args.feature_set == "extended":
            factory = ExtendedFeatureFactory(use_extended=True)
        else:
            factory = SMCFeatureFactory()
    """

    def __init__(self, swing_length: int = 5, use_extended: bool | None = None):
        self._base = SMCFeatureFactory(swing_length=swing_length)
        if use_extended is None:
            use_extended = _use_extended_default()
        self.use_extended = use_extended

    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run base features, then optionally append extended columns."""
        df = self._base.compute_features(df)
        if df.is_empty():
            return df
        if self.use_extended:
            df = self._append_4h_features(df)
            df = self._append_microstructure_features(df)
        return df

    # ── 4H / Macro features ────────────────────────────────────────────────

    def _append_4h_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds four macro/trend features that carry signal at coarser timeframes.
        All use trailing windows — zero lookahead.
        """
        eps = 1e-12

        # 1. trend_strength: ratio of short-window ATR to long-window ATR.
        #    > 1.0 = volatility expanding (trending), < 1.0 = volatility contracting.
        true_range_col = "true_range"
        if true_range_col not in df.columns:
            # Reconstruct if base factory dropped it
            tr = (
                pl.max_horizontal([
                    pl.col("high") - pl.col("low"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low")  - pl.col("close").shift(1)).abs(),
                ])
            )
            df = df.with_columns(tr.alias(true_range_col))

        atr_fast = pl.col(true_range_col).rolling_mean(window_size=7,  min_samples=7)
        atr_slow = pl.col(true_range_col).rolling_mean(window_size=28, min_samples=28)
        trend_strength = (atr_fast / (atr_slow + eps)).fill_nan(1.0).fill_null(1.0)

        # 2. momentum_slope: normalised close slope over 10 bars.
        #    Approximates linear-regression slope without a custom UDF by using
        #    the diff of a 10-bar rolling mean (LWMA-like proxy).
        close_mean_10 = pl.col("close").rolling_mean(window_size=10, min_samples=10)
        close_mean_5  = pl.col("close").rolling_mean(window_size=5,  min_samples=5)
        momentum_slope = (
            (close_mean_5 - close_mean_10) / (pl.col("close").clip(lower_bound=eps))
        ).fill_nan(0.0).fill_null(0.0)

        # 3. vol_regime: short-window vol / long-window vol.
        #    Values > 1 signal elevated volatility relative to history.
        vol_short = pl.col("close").pct_change().rolling_std(window_size=20, min_samples=20)
        vol_long  = pl.col("close").pct_change().rolling_std(window_size=100, min_samples=100)
        vol_regime = (vol_short / (vol_long + eps)).fill_nan(1.0).fill_null(1.0)

        # 4. candle_body_ratio: |open - close| / (high - low + eps).
        #    Near 0 → doji/chop; near 1 → strong directional candle.
        body = (pl.col("close") - pl.col("open")).abs()
        rng  = (pl.col("high") - pl.col("low")).clip(lower_bound=eps)
        candle_body_ratio = (body / rng).fill_nan(0.0).fill_null(0.0)

        df = df.with_columns([
            trend_strength.alias("trend_strength"),
            momentum_slope.alias("momentum_slope"),
            vol_regime.alias("vol_regime"),
            candle_body_ratio.alias("candle_body_ratio"),
        ])

        # Clamp extended features to [-5, 5] to prevent outlier explosions
        ext_cols = ["trend_strength", "momentum_slope", "vol_regime", "candle_body_ratio"]
        df = df.with_columns([
            pl.col(c).clip(lower_bound=-5.0, upper_bound=5.0) for c in ext_cols
        ])

        return df

    # ── Microstructure / OBI features ─────────────────────────────────────

    def _append_microstructure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Appends OBI and volume delta columns IF they exist in the DataFrame schema.
        Silently skips any column that is not present — this factory works on plain
        OHLCV data and upgrades automatically when richer data is supplied.

        Expected optional columns (added by data_harvester OBI pipeline):
            bid_vol, ask_vol  → obi           (order book imbalance)
            up_vol, dn_vol    → volume_delta   (signed volume delta)
            funding_rate      → funding_rate   (perpetual funding rate)
        """
        eps = 1e-12
        new_cols = []

        # 1. Order Book Imbalance
        if "bid_vol" in df.columns and "ask_vol" in df.columns:
            bid = pl.col("bid_vol").cast(pl.Float32)
            ask = pl.col("ask_vol").cast(pl.Float32)
            obi = ((bid - ask) / (bid + ask + eps)).fill_nan(0.0).fill_null(0.0)
            new_cols.append(obi.alias("obi"))
        else:
            # Placeholder so target_dim is stable even without OBI data
            new_cols.append(pl.lit(0.0).cast(pl.Float32).alias("obi"))

        # 2. Volume Delta
        if "up_vol" in df.columns and "dn_vol" in df.columns:
            up = pl.col("up_vol").cast(pl.Float32)
            dn = pl.col("dn_vol").cast(pl.Float32)
            total = (up + dn).clip(lower_bound=eps)
            vd = ((up - dn) / total).fill_nan(0.0).fill_null(0.0)
            new_cols.append(vd.alias("volume_delta"))
        else:
            new_cols.append(pl.lit(0.0).cast(pl.Float32).alias("volume_delta"))

        # 3. Funding Rate
        if "funding_rate" in df.columns:
            fr = pl.col("funding_rate").cast(pl.Float32).fill_nan(0.0).fill_null(0.0)
            new_cols.append(fr.alias("funding_rate"))
        else:
            new_cols.append(pl.lit(0.0).cast(pl.Float32).alias("funding_rate"))

        if new_cols:
            df = df.with_columns(new_cols)

        return df


# ── Column-count helper ────────────────────────────────────────────────────────

def count_feature_columns(
    df: pl.DataFrame,
    use_extended: bool = False,
) -> int:
    """
    Returns the number of numeric feature columns that would be extracted from
    a processed DataFrame (mirrors the exclude_cols logic in train.py).
    Useful for setting target_dim in KellyConvexEnv dynamically.
    """
    exclude_cols = {"open", "high", "low", "close", "volume", "true_range",
                    "timestamp", "is_swing_high", "is_swing_low"}
    numeric_types = {pl.Float32, pl.Float64, pl.Int32, pl.Int64}
    return sum(
        1 for col, dtype in df.schema.items()
        if dtype in numeric_types and col not in exclude_cols
    )
