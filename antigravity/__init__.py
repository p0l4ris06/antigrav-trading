"""
Antigravity — Autonomous RL-driven quantitative trading system.

Subsystems:
    antigravity.db          ClickHouse tick persistence
    antigravity.gateway     FastAPI WebSocket gateway
    antigravity.features    Polars-based feature engineering
    antigravity.regime      PCA → GMM regime classifier
    antigravity.rl          PPO/SAC reinforcement learning core
    antigravity.overseer    Agentic self-healing daemon
    antigravity.export      ONNX serialization utilities
"""

__version__ = "0.1.0"
