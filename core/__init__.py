# core/__init__.py
# ─── Public API for the core computation package ─────────────────────────────
#
# Directory map:
#   agent.py            — KellyConvexEnv (Gymnasium env), init_agent(), load_agent_model()
#   features.py         — SMCFeatureFactory (Polars OHLCV → RL state vector, base feature set)
#   features_extended.py— ExtendedFeatureFactory (4H + OBI augmentation, togglable — Phase 1)
#   alpaca_bridge.py    — AlpacaQuantBridge (live data + order execution via Alpaca)
#   exchange_adapter.py — OmniGateway / RiskManager (ccxt multi-exchange, used by live_daemon.py)
#                         [formerly core/gateway.py]
#   drift_detector.py   — PageHinkleyDrift, Daemon (regime-shift detection)
#                         [formerly core/overseer.py]
#   backtester.py       — AntigravBacktester (offline replay against saved model)
#
# Web/daemon infrastructure lives under antigravity/:
#   antigravity/gateway/server.py  — FastAPI WebSocket gateway (NOT the ccxt gateway above)
#   antigravity/overseer/daemon.py — Agentic self-healing overseer (NOT the drift detector above)
#   antigravity/regime/classifier.py — PCA + GMM regime classification
# ─────────────────────────────────────────────────────────────────────────────

from core.features import SMCFeatureFactory
from core.agent import KellyConvexEnv, init_agent, load_agent_model

# NOTE: AlpacaQuantBridge is intentionally NOT imported here.
# alpaca-py is a live-trading SDK that may not be installed in the training
# environment. Import it directly where needed:
#   from core.alpaca_bridge import AlpacaQuantBridge

__all__ = [
    "SMCFeatureFactory",
    "KellyConvexEnv",
    "init_agent",
    "load_agent_model",
]
