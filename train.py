"""
train.py — Antigravity Training & Evaluation Pipeline
======================================================
Trains the Kelly-Convex RL trading agent and performs Walk-Forward OOS evaluation.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
import numpy as np
import polars as pl
import gymnasium as gym

# Inject local core directory into path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.features import SMCFeatureFactory
from core.agent import KellyConvexEnv, init_agent


def generate_synthetic_data(n_rows: int = 2000) -> pl.DataFrame:
    """Generate realistic synthetic OHLCV data for robust training fallbacks."""
    np.random.seed(42)
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n_rows)]
    close = 50000.0 + np.cumsum(np.random.normal(0, 100, n_rows))
    high = close + np.abs(np.random.normal(10, 5, n_rows))
    low = close - np.abs(np.random.normal(10, 5, n_rows))
    open_val = close - np.random.normal(0, 5, n_rows)
    return pl.DataFrame({
        "timestamp": dates,
        "open": open_val,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.uniform(1, 10, n_rows)
    })


def evaluate(model, env):
    """Run model deterministically and return final portfolio value."""
    from stable_baselines3.common.vec_env import VecEnv
    is_vec = isinstance(env, VecEnv)
    
    if is_vec:
        obs = env.reset()
    else:
        obs, _ = env.reset()
        
    done = False
    portfolio_value = 1.0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if is_vec:
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            portfolio_value = infos[0].get("portfolio_value", portfolio_value)
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            portfolio_value = info.get("portfolio_value", portfolio_value)
            
    return portfolio_value


def main():
    parser = argparse.ArgumentParser(description="Antigravity Training Pipeline")
    parser.add_argument("--data", type=str, nargs="+", default=["data/BTC_USDT_15m.parquet"], help="Path to data file(s) or directories")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")
    args = parser.parse_args()

    # 1. Load Data — supports multiple files/directories of parquets
    paths = args.data
    parquet_files = []
    for path in paths:
        if os.path.isdir(path):
            p_files = sorted([
                os.path.join(path, f)
                for f in os.listdir(path) if f.endswith(".parquet")
            ])
            parquet_files.extend(p_files)
        elif os.path.exists(path) and path.endswith(".parquet"):
            parquet_files.append(path)

    if not parquet_files:
        print(f"No parquet files found in {paths} — falling back to synthetic data.")
        df = generate_synthetic_data()
    else:
        frames = []
        for pf in parquet_files:
            try:
                part = pl.read_parquet(pf)
                if "timestamp" in part.columns:
                    frames.append(part)
                else:
                    print(f"Skipping {pf} — no timestamp column.")
            except Exception as e:
                print(f"Error reading {pf}: {e}")
        if frames:
            # Sort each individual asset frame by timestamp, then concatenate sequentially
            # to prevent multi-asset interleaving and concurrent co-temporal look-ahead leaks.
            sorted_frames = [f.sort("timestamp") for f in frames]
            df = pl.concat(sorted_frames, how="vertical_relaxed")
            # Ensure timestamp is parsed as Datetime so it does not pollute the numeric features
            if "timestamp" in df.columns:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
            print(f"Loaded {len(frames)} parquet files -> {df.height:,} total rows")
        else:
            df = generate_synthetic_data()

    # 2. Feature Engineering
    factory = SMCFeatureFactory()
    features_df = factory.compute_features(df)

    # Convert to numeric features array, excluding raw prices / non-stationary inputs
    exclude_cols = {"open", "high", "low", "close", "volume", "true_range"}
    numeric_cols = [c for c, t in features_df.schema.items() if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and c not in exclude_cols]
    features_np = features_df.select(numeric_cols).to_numpy().astype(np.float32)

    # Dynamic padding/truncation to ensure compatibility with KellyConvexEnv shape=(9,)
    target_dim = 9
    if features_np.shape[1] < target_dim:
        padding = np.zeros((features_np.shape[0], target_dim - features_np.shape[1]), dtype=np.float32)
        features_np = np.hstack([features_np, padding])
    elif features_np.shape[1] > target_dim:
        features_np = features_np[:, :target_dim]

    # Clean NaN/inf
    features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # HARD CLIP: Constraints features to sane bounds for PPO stability
    features_np = np.clip(features_np, -1e3, 1e3)

    # 3. Train/Test Split (Walk-Forward Chronological Split)
    split_idx = int(len(features_np) * 0.75)  # Train 18 months, Eval 6 months equivalent
    train_data = features_np[:split_idx]
    test_data = features_np[split_idx:]

    # 4. Agent Training
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    train_vec_env = DummyVecEnv([lambda: KellyConvexEnv(data_stream=train_data, max_leverage=3.0, target_dim=9)])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model, _ = init_agent(target_dim=9)
    # Re-bind environment to train_vec_env
    model.set_env(train_vec_env)
    model.learn(total_timesteps=args.timesteps)

    # 4. Walk-Forward Evaluation (Out-Of-Sample)
    test_vec_env = DummyVecEnv([lambda: KellyConvexEnv(data_stream=test_data, max_leverage=3.0, max_episode_steps=len(test_data), target_dim=9)])
    test_vec_env = VecNormalize(test_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Sync normalization stats from training
    from copy import deepcopy
    test_vec_env.obs_rms = deepcopy(train_vec_env.obs_rms)
    test_vec_env.ret_rms = deepcopy(train_vec_env.ret_rms)
    test_vec_env.training = False

    final_oos_wealth = evaluate(model, test_vec_env)

    # --- NEW: SAVE THE MODEL FOR LIVE TRADING ---
    os.makedirs("models", exist_ok=True)
    # Save model weights
    model.save("models/ppo_antigrav_latest") 
    # Save the VecNormalize stats (under models/vec_normalize.pkl)
    train_vec_env.save("models/vec_normalize.pkl")
    
    # 5. The Critical Output for the Autoresearcher
    print(f"FITNESS_SCORE: {final_oos_wealth:.4f}")


if __name__ == "__main__":
    main()
