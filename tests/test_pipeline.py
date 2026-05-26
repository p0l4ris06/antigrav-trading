import pytest
import numpy as np
import polars as pl
import gymnasium as gym
from core.features import SMCFeatureFactory
from core.agent import KellyConvexEnv, init_agent


def generate_mock_ohlcv(n_rows: int = 100) -> pl.DataFrame:
    """Generate mock OHLCV dataset for testing."""
    import datetime
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.normal(0, 1.0, n_rows))
    high = close + np.abs(np.random.normal(0.5, 0.2, n_rows))
    low = close - np.abs(np.random.normal(0.5, 0.2, n_rows))
    open_val = close - np.random.normal(0, 0.3, n_rows)
    volume = np.random.uniform(10, 100, n_rows)
    
    start_date = datetime.datetime(2026, 5, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)
    dates = [start_date + datetime.timedelta(minutes=15 * i) for i in range(n_rows)]
    
    return pl.DataFrame({
        "timestamp": dates,
        "open": open_val.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.float32),
    })


def test_feature_factory_basics():
    """Verify basic characteristics of the SMCFeatureFactory."""
    df = generate_mock_ohlcv(100)
    factory = SMCFeatureFactory(swing_length=2)
    features_df = factory.compute_features(df)
    
    # Check that output is a polars DataFrame
    assert isinstance(features_df, pl.DataFrame)
    
    # Check that empty DataFrame returns empty DataFrame instead of crashing
    empty_df = pl.DataFrame(schema=df.schema)
    assert factory.compute_features(empty_df).is_empty()


def test_look_ahead_bias():
    """
    Look-ahead bias check:
    Modifying future prices must not affect historical feature values.
    """
    df = generate_mock_ohlcv(100)
    factory = SMCFeatureFactory(swing_length=3)
    
    # Compute baseline features
    features_baseline = factory.compute_features(df)
    
    # We choose an index to inspect (e.g. 50 in input df)
    inspect_idx = 50
    inspect_ts = df["timestamp"][inspect_idx]
    
    # Create a copy and mutate prices AFTER inspect_idx
    df_mutated = df.clone()
    
    # Mutate close, high, low, open for indices > inspect_idx
    mutated_close = df_mutated["close"].to_numpy().copy()
    mutated_close[inspect_idx + 1:] = mutated_close[inspect_idx + 1:] * 2.0
    
    mutated_high = df_mutated["high"].to_numpy().copy()
    mutated_high[inspect_idx + 1:] = mutated_high[inspect_idx + 1:] * 2.0
    
    mutated_low = df_mutated["low"].to_numpy().copy()
    mutated_low[inspect_idx + 1:] = mutated_low[inspect_idx + 1:] / 2.0
    
    df_mutated = df_mutated.with_columns([
        pl.Series("close", mutated_close),
        pl.Series("high", mutated_high),
        pl.Series("low", mutated_low),
    ])
    
    # Recompute features on mutated data
    features_mutated = factory.compute_features(df_mutated)
    
    # Find the matching rows by timestamp
    baseline_row = features_baseline.filter(pl.col("timestamp") == inspect_ts).row(0, named=True)
    mutated_row = features_mutated.filter(pl.col("timestamp") == inspect_ts).row(0, named=True)
    
    # Verify that all features up to inspect_idx are identical
    for col in baseline_row.keys():
        if col == "timestamp":
            continue
        # Allow small floating point precision differences
        assert pytest.approx(baseline_row[col], abs=1e-5) == mutated_row[col], (
            f"Look-ahead bias detected in feature: {col}!"
        )


def test_env_dimensions_and_step():
    """Verify that KellyConvexEnv matches target dimensions and steps correctly."""
    # Generate mock features matching the pipeline format
    np.random.seed(42)
    data = np.random.normal(0, 1.0, (100, 9)).astype(np.float32)
    
    # Initialize environment
    env = KellyConvexEnv(data_stream=data, max_leverage=3.0, target_dim=9)
    
    obs, info = env.reset()
    assert obs.shape == (9,)
    assert isinstance(info, dict)
    
    # Take a step with a valid action: [bias, kelly_allocation]
    # action[0] is bias in [-1, 1], action[1] is kelly fraction in [0, 1]
    action = np.array([0.5, 0.1], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs.shape == (9,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "portfolio_value" in info


def test_agent_initialization():
    """Verify that init_agent returns working PPO model and env."""
    model, env = init_agent(target_dim=9)
    assert model is not None
    assert env is not None
    assert env.observation_space.shape == (9,)
