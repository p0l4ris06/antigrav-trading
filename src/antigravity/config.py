"""
Centralized configuration via Pydantic BaseSettings.

All values are overridable via environment variables prefixed with AG_.
See .env.example for the complete parameter reference.
"""

from __future__ import annotations
import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AccountConfig(BaseSettings):
    """Exchange account credentials."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_ACCOUNT_", extra="ignore")

    api_key: str = ""
    api_secret: str = ""
    subaccount: str | None = None


class ClickHouseConfig(BaseSettings):
    """ClickHouse connection parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_CH_", extra="ignore")

    host: str = "localhost"
    port: int = 8123
    user: str = "admin"
    password: str = "admin123"
    database: str = "antigravity"


class RLConfig(BaseSettings):
    """Reinforcement learning hyperparameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_RL_", extra="ignore")

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    lambda_penalty: float = Field(
        default=1.5,
        description="Asymmetric MAE penalty scalar λ in R_t = (MFE - λ·MAE) / ATR",
    )
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    net_arch_pi: list[int] = [256, 256]
    net_arch_vf: list[int] = [256, 256]


class LatencyConfig(BaseSettings):
    """Predictive latency modeling parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_LATENCY_", extra="ignore")

    enabled: bool = False
    history_len: int = 20  # k steps in [s_{t-k}, ..., s_t]
    hidden_dim: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    train_interval_steps: int = 1000


class OverseerConfig(BaseSettings):
    """Agentic overseer parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_OVERSEER_", extra="ignore")

    sharpe_window: int = Field(
        default=2880,
        description="Rolling Sharpe window in 1-min bars (48h = 2880)",
    )
    drift_delta: float = Field(
        default=0.005,
        description="Page-Hinkley minimum magnitude of allowed change",
    )
    drift_threshold: float = Field(
        default=50.0,
        description="Page-Hinkley detection threshold",
    )
    refit_percentile: float = Field(
        default=5.0,
        description="Log-likelihood percentile below which GMM refit triggers",
    )
    swap_p_value: float = Field(
        default=0.05,
        description="Paired t-test significance threshold for policy hot-swap",
    )
    shadow_train_timesteps: int = Field(
        default=50_000,
        description="Training timesteps for the shadow fork policy",
    )
    proactive_shadow: bool = Field(
        default=False,
        description="If True, start a shadow fork immediately without waiting for drift",
    )


class FeatureConfig(BaseSettings):
    """Feature factory parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_FF_", extra="ignore")

    correlation_threshold: float = Field(
        default=0.85,
        description="Spearman rank correlation threshold for feature pruning",
    )
    atr_period: int = 14
    tick_buffer_size: int = 50_000
    vwap_windows: list[int] = [300, 900, 3600]  # seconds: 5m, 15m, 1h
    rolling_vol_windows: list[int] = [20, 50]


class ExchangeConfig(BaseSettings):
    """Exchange adapter parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_EXCHANGE_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable live exchange WebSocket feed (disable for simulated mode)",
    )
    adapter: str = Field(
        default="binance",
        description="Exchange adapter to use: 'binance'",
    )
    symbol: str = "BTCUSDT"
    symbols: list[str] = ["BTCUSDT"]
    futures: bool = False


class ExecutionConfig(BaseSettings):
    """Live trading execution parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_EXECUTION_", extra="ignore")

    enabled: bool = False
    max_position_size: float = 0.1  # Fraction of portfolio
    leverage: int = 1
    slippage_tolerance: float = 0.001


class GatewayConfig(BaseSettings):
    """FastAPI gateway parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_GATEWAY_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    ws_queue_size: int = 500_000
    batch_size: int = 1000
    batch_flush_interval_ms: int = 500


class TelemetryConfig(BaseSettings):
    """OpenTelemetry / LangSmith tracing parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_OTEL_", extra="ignore")

    enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing export to LangSmith",
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com/otel",
        description="OTLP HTTP endpoint for trace export",
    )
    api_key: str = Field(
        default="",
        description="LangSmith API key (x-api-key header)",
    )
    project_name: str = Field(
        default="antigravity",
        description="LangSmith project name for trace grouping",
    )
    service_name: str = Field(
        default="antigravity-trading",
        description="OTel service name",
    )
    sample_rate: float = Field(
        default=1.0,
        description="Trace sampling rate (0.0-1.0). Set <1.0 to reduce volume.",
    )


class DataConfig(BaseSettings):
    """Data Pipeline storage parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_DATA_", extra="ignore")

    raw_dir: Path = Field(default_factory=lambda: Path(os.getenv("AG_DATA_DIR", "data")) / "raw")
    processed_dir: Path = Field(default_factory=lambda: Path(os.getenv("AG_DATA_DIR", "data")) / "processed")
    retention_days: int = 365


class ModelConfig(BaseSettings):
    """Model storage & cache parameters."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_MODEL_", extra="ignore")

    registry_dir: Path = Field(default_factory=lambda: Path(os.getenv("AG_MODEL_DIR", "models")))
    ppo_latest: Path = Field(default_factory=lambda: Path(os.getenv("AG_MODEL_DIR", "models")) / "ppo_antigrav_latest.zip")
    regime_classifier: Path = Field(default_factory=lambda: Path(os.getenv("AG_MODEL_DIR", "models")) / "regime_classifier.pkl")
    cache_size_mb: int = 500


class AntigravityConfig(BaseSettings):
    """Root configuration aggregating all subsystem configs."""
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AG_", extra="ignore")

    clickhouse: ClickHouseConfig = ClickHouseConfig()
    rl: RLConfig = RLConfig()
    overseer: OverseerConfig = OverseerConfig()
    features: FeatureConfig = FeatureConfig()
    gateway: GatewayConfig = GatewayConfig()
    exchange: ExchangeConfig = ExchangeConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    latency: LatencyConfig = LatencyConfig()
    account: AccountConfig = AccountConfig()
    execution: ExecutionConfig = ExecutionConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()


# Module-level singleton — import this from anywhere
settings = AntigravityConfig()
