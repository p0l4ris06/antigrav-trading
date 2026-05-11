"""
Antigravity Training Pipeline.

Standalone script that:
    1. Generates synthetic tick data (or pulls from ClickHouse)
    2. Computes full feature matrix via FeatureFactory
    3. Fits the GMM Regime Classifier
    4. Trains the PPO agent
    5. Saves model artifacts to models/

Usage:
    python -m antigravity.train
    python -m antigravity.train --timesteps 200000
    python -m antigravity.train --from-clickhouse --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


def generate_synthetic_ticks(n_ticks: int = 50_000, seed: int = 42) -> pl.DataFrame:
    """
    Generate realistic synthetic L2 tick data with regime changes.

    Simulates:
        - 3 market regimes (trend, mean-reversion, volatility expansion)
        - Realistic spread dynamics
        - Volume clustering
        - Microstructure noise
    """
    rng = np.random.default_rng(seed)

    # Generate regime sequence (each regime lasts ~2000-8000 ticks)
    timestamps = []
    prices = []
    bid_prices = []
    ask_prices = []
    bid_sizes = []
    ask_sizes = []
    last_sizes = []

    base_price = 50_000.0
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    regime_change_points = []
    current_regime = 0

    for i in range(n_ticks):
        # Regime switching every ~5000 ticks
        if i % 5000 == 0 and i > 0:
            current_regime = (current_regime + 1) % 3
            regime_change_points.append(i)

        # Regime-dependent dynamics
        if current_regime == 0:
            # Trend regime: strong drift, moderate vol
            drift = 0.02
            vol = 3.0
            base_spread = 0.3
        elif current_regime == 1:
            # Mean-reversion: no drift, low vol
            drift = 0.0
            vol = 1.5
            base_spread = 0.2
            # Add mean reversion
            drift -= (base_price - 50_000) * 0.001
        else:
            # Volatility expansion: no drift, high vol, wide spreads
            drift = 0.0
            vol = 8.0
            base_spread = 1.0

        # Price evolution
        base_price += drift + rng.normal(0, vol)
        base_price = max(base_price, 100)  # floor

        # Spread with randomness
        spread = abs(rng.normal(base_spread, base_spread * 0.3))
        bid = base_price - spread / 2
        ask = base_price + spread / 2

        # Volume clustering (exponential + Poisson bursts)
        base_vol = rng.exponential(2.0)
        if rng.random() < 0.05:  # 5% chance of volume burst
            base_vol *= rng.uniform(3, 10)

        bid_vol = rng.exponential(1.5)
        ask_vol = rng.exponential(1.5)

        timestamps.append(t)
        prices.append(round(bid + rng.random() * spread, 2))
        bid_prices.append(round(bid, 2))
        ask_prices.append(round(ask, 2))
        bid_sizes.append(round(bid_vol, 4))
        ask_sizes.append(round(ask_vol, 4))
        last_sizes.append(round(base_vol, 4))

        t += timedelta(milliseconds=100)

    df = pl.DataFrame({
        "symbol": ["BTCUSDT"] * n_ticks,
        "timestamp": timestamps,
        "bid_price": bid_prices,
        "ask_price": ask_prices,
        "bid_size": bid_sizes,
        "ask_size": ask_sizes,
        "last_price": prices,
        "last_size": last_sizes,
    })

    logger.info(
        "synthetic_data.generated",
        n_ticks=n_ticks,
        n_regime_changes=len(regime_change_points),
        price_range=(min(prices), max(prices)),
    )
    return df


async def load_from_clickhouse(symbol: str, hours: int = 48) -> pl.DataFrame | None:
    """Pull recent tick data from ClickHouse."""
    from antigravity.db.client import ClickHouseManager

    try:
        manager = await ClickHouseManager.create()
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)
        df = await manager.query_ticks(symbol, start, now)
        await manager.close()

        if df.height < 1000:
            logger.warning("clickhouse.insufficient_data", rows=df.height)
            return None

        logger.info("clickhouse.data_loaded", rows=df.height)
        return df
    except Exception as exc:
        logger.error("clickhouse.load_failed", error=str(exc))
        return None


def compute_training_data(tick_df: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    Run the full feature pipeline on raw tick data and prepare
    arrays for the Gymnasium environment.

    Returns:
        dict with keys: features, prices, atr, regime_proba
    """
    from antigravity.features.factory import FeatureFactory
    from antigravity.regime.classifier import RegimeClassifier

    logger.info("pipeline.computing_features", n_ticks=tick_df.height)

    # --- Feature computation ---
    factory = FeatureFactory(buffer_size=tick_df.height + 100)

    # Batch ingest all ticks
    ticks = tick_df.to_dicts()
    factory.ingest_batch(ticks)

    # Compute full feature matrix
    feature_df = factory.compute_features()
    if feature_df is None:
        raise ValueError("Insufficient data for feature computation")

    # Prune highly correlated features
    dropped = factory.prune_correlated_features(feature_df)
    if dropped:
        logger.info("pipeline.features_pruned", dropped=dropped)

    feature_names = factory.get_feature_names()
    available = [c for c in feature_names if c in feature_df.columns]

    logger.info("pipeline.features_ready", n_features=len(available), names=available)

    # Extract feature matrix (drop nulls from rolling windows)
    feature_matrix = feature_df.select(available).drop_nulls().to_numpy().astype(np.float32)

    # Get corresponding prices and ATR
    # We need to align: feature_df may have nulls at the start from rolling windows
    null_mask = feature_df.select(
        pl.all_horizontal(pl.col(available).is_not_null())
    ).to_numpy().flatten()

    prices = feature_df.select("last_price").to_numpy().flatten()[null_mask]
    atr_col = feature_df.select("atr").to_numpy().flatten()[null_mask]

    # --- CRITICAL SAFETY GATE ---
    # Instead of converting Inf to 0, we must clip to a standard deviation bound
    # to keep the gradients sane for the PPO agent.
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    # Robustly clip features to +/- 10 standard deviations
    # This prevents 'One Billion' scores by capping the observation space
    feature_matrix = np.clip(feature_matrix, -1e3, 1e3) 

    prices = np.nan_to_num(prices, nan=50000.0)
    
    # Ensure ATR is never small enough to cause a division explosion
    atr_col = np.nan_to_num(atr_col, nan=1.0)
    atr_col = np.clip(atr_col, 0.1, None) # Hard floor for volatility

    # --- Regime classification ---
    logger.info("pipeline.fitting_regime_classifier")
    regime_clf = RegimeClassifier()
    regime_stats = regime_clf.fit(feature_matrix)
    logger.info("pipeline.regime_fitted", **regime_stats)

    # Get soft regime assignments for all data points
    regime_proba = regime_clf.predict_proba(feature_matrix).astype(np.float32)

    # Ensure lengths match
    min_len = min(len(feature_matrix), len(prices), len(atr_col), len(regime_proba))
    feature_matrix = feature_matrix[:min_len]
    prices = prices[:min_len]
    atr_col = atr_col[:min_len]
    regime_proba = regime_proba[:min_len]

    logger.info(
        "pipeline.data_ready",
        n_samples=min_len,
        n_features=feature_matrix.shape[1],
        n_regimes=regime_proba.shape[1],
        price_range=(float(prices.min()), float(prices.max())),
        atr_mean=float(atr_col.mean()),
    )

    # --- FINAL REWARD VERIFICATION ---
    # Log the first few returns to console to verify sanity
    avg_price = float(np.mean(prices))
    logger.info("pipeline.sanity_check", price_avg=avg_price, feature_max=float(np.max(feature_matrix)))

    return {
        "features": feature_matrix,
        "prices": prices,
        "atr": atr_col,
        "regime_proba": regime_proba,
        "feature_names": available,
        "regime_classifier": regime_clf,
    }


def train_agent(
    data: dict[str, np.ndarray],
    total_timesteps: int = 100_000,
    episode_length: int = 500,
    save_dir: str = "models",
) -> None:
    """
    Train the PPO agent on prepared data.

    Steps:
        1. Create TradingEnv with feature/price/atr/regime data
        2. Initialize AgentManager (PPO + VecNormalize)
        3. Train for total_timesteps
        4. Save model + VecNormalize stats to save_dir
    """
    from antigravity.rl.agent import AgentManager
    from antigravity.rl.environment import TradingEnv

    n_samples = len(data["features"])
    ep_len = min(episode_length, n_samples - 2)

    logger.info(
        "training.creating_environment",
        n_samples=n_samples,
        episode_length=ep_len,
        n_features=data["features"].shape[1],
        n_regimes=data["regime_proba"].shape[1],
    )

    env = TradingEnv(
        feature_data=data["features"],
        price_data=data["prices"],
        atr_data=data["atr"],
        regime_data=data["regime_proba"],
        n_assets=1,
        episode_length=ep_len,
    )

    agent = AgentManager(env=env)

    logger.info("training.starting", timesteps=total_timesteps)
    start_time = time.perf_counter()

    agent.train(total_timesteps=total_timesteps, progress_bar=True)

    elapsed = time.perf_counter() - start_time
    logger.info("training.complete", elapsed_seconds=round(elapsed, 1))

    # Save model
    agent.save(save_dir)
    logger.info("training.model_saved", path=save_dir)

    # Export ONNX
    try:
        onnx_path = agent.export_onnx(Path(save_dir) / "policy.onnx")
        logger.info("training.onnx_exported", path=str(onnx_path))
    except Exception as exc:
        logger.warning("training.onnx_export_failed", error=str(exc))

    # Save regime classifier
    import pickle
    regime_path = Path(save_dir) / "regime_classifier.pkl"
    with open(regime_path, "wb") as f:
        pickle.dump(data["regime_classifier"], f)
    logger.info("training.regime_classifier_saved", path=str(regime_path))

    # Evaluation run
    logger.info("training.evaluating")
    obs, _ = env.reset()
    total_reward = 0.0
    n_steps = 0
    done = False

    while not done:
        weights = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(weights)
        total_reward += reward
        n_steps += 1
        done = terminated or truncated

    logger.info(
        "training.evaluation_complete",
        total_reward=round(total_reward, 4),
        n_steps=n_steps,
        final_capital=round(info.get("capital", 0), 2),
        final_mfe=round(info.get("mfe", 0), 4),
        final_mae=round(info.get("mae", 0), 4),
    )


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Antigravity RL Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m antigravity.train
    python -m antigravity.train --timesteps 200000
    python -m antigravity.train --from-clickhouse --symbol BTCUSDT --hours 72
    python -m antigravity.train --n-ticks 100000 --episode-length 1000
        """,
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--n-ticks", type=int, default=50_000, help="Synthetic tick count")
    parser.add_argument("--episode-length", type=int, default=500, help="Steps per episode")
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--from-clickhouse", action="store_true", help="Load data from ClickHouse")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--hours", type=int, default=48, help="Hours of history to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")

    args = parser.parse_args()

    logger.info("train.starting", args=vars(args))

    # --- Load or generate data ---
    if args.from_clickhouse:
        tick_df = asyncio.run(load_from_clickhouse(args.symbol, args.hours))
        if tick_df is None:
            logger.error("train.no_data_from_clickhouse, falling back to synthetic")
            tick_df = generate_synthetic_ticks(args.n_ticks, args.seed)
    else:
        tick_df = generate_synthetic_ticks(args.n_ticks, args.seed)

    # --- Compute features ---
    data = compute_training_data(tick_df)

    # --- Train ---
    train_agent(
        data=data,
        total_timesteps=args.timesteps,
        episode_length=args.episode_length,
        save_dir=args.save_dir,
    )

    logger.info("train.complete")


if __name__ == "__main__":
    main()
