# ANTIGRAV TRADING - Backend Setup Guide

## Project Structure

```
C:\Users\Wren C\Documents\ANTIGRAV TRADING\
├── pyproject.toml                    # Project config (requires Python 3.11+)
├── requirements.txt                  # Alternative dependencies list
├── antigravity/                      # Main package (CORRECT ENTRY POINT)
│   ├── __init__.py
│   ├── config.py                    # Settings management
│   ├── gateway/
│   │   ├── server.py                # ⭐ MAIN ENTRY POINT (FastAPI app)
│   │   └── ws_consumer.py
│   ├── features/
│   │   └── factory.py               # Feature computation pipeline
│   ├── db/
│   │   └── client.py                # ClickHouse persistence
│   ├── overseer/
│   │   └── daemon.py                # Agentic overseer
│   ├── regime/
│   │   └── classifier.py            # Market regime detection
│   ├── rl/                          # Reinforcement learning agent
│   └── tracing.py                   # OpenTelemetry integration
├── core/                             # Legacy/alternative modules
│   ├── gateway.py
│   ├── features.py
│   └── ...
├── models/                           # Pre-trained model storage
│   ├── ppo_model.zip               # RL policy (optional)
│   ├── regime_classifier.pkl       # Regime model (optional)
│   └── ...
└── dashboard/                        # Frontend React app (optional)
    └── dist/                         # Built static files
```

## Requirements

- **Python 3.11+** (strictly required per pyproject.toml)
- **pip** (for dependency installation)
- Virtual environment support

## Installation Steps

### 1. Check Python Version
```cmd
python --version
```
Expected: Python 3.11.x or higher

### 2. Create Virtual Environment
```cmd
cd "C:\Users\Wren C\Documents\ANTIGRAV TRADING"
python -m venv venv
```

### 3. Activate Virtual Environment
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` prefix in your command prompt.

### 4. Install Dependencies
```cmd
pip install -e .
```

This reads from `pyproject.toml` and installs:
- **FastAPI 0.115+** - Web framework
- **Uvicorn 0.30+** - ASGI server
- **Polars 1.0+** - High-performance data processing
- **PyTorch 2.4+** - ML/RL models
- **Stable-Baselines3 2.4+** - RL algorithms
- **ClickHouse Python client** - Time-series database
- **Pydantic 2.9+** - Data validation
- **And 25+ more dependencies**

### 5. Start the Backend Server

```cmd
python -m uvicorn antigravity.gateway.server:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 [Press ENTER to quit]
INFO:     gateway.starting
INFO:     event_loop.winloop_installed  # (or: winloop_unavailable, using default)
INFO:     gateway.feature_factory_ready active_features=[...]
INFO:     gateway.regime_classifier_ready fitted=False
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Available Endpoints

### REST API
- `GET /api/status` - System health and runtime statistics
- `GET /api/prices` - Current prices for all symbols
- `POST /api/control/execution` - Enable/disable trading
- `POST /api/control/account` - Update exchange credentials
- `POST /api/control/action` - Trigger control actions (pause, resume, retrain, etc.)

### WebSocket
- `WS /ws` - Real-time tick ingestion and system updates

### Dashboard
- `GET /` - Serves the React dashboard (if `dashboard/dist/` exists)

## What Happens on Startup

1. **Event Loop** - Installs `winloop` (Windows) or `uvloop` (Linux) for high performance
2. **OpenTelemetry** - Initializes tracing (optional, for LangSmith export)
3. **ClickHouse Connection** - Attempts to connect to persistent database
   - Falls back gracefully if unavailable
4. **Feature Factory** - Loads and configures feature computation pipeline
5. **Regime Classifier** - Loads pre-trained regime detector (if exists at `models/regime_classifier.pkl`)
6. **RL Agent** - Loads pre-trained policy (if exists at `models/ppo_model.zip`)
7. **Agentic Overseer** - Initializes monitoring daemon
8. **Auto-Simulation** - Starts synthetic tick generation for testing
9. **Dashboard** - Mounts React frontend (if built)

## Key Features

### Hot-Path Data Processing
- **Queue-based ingestion** - Async WebSocket with configurable queue depth
- **Vectorized computation** - Polars DataFrames for SIMD operations
- **Backpressure handling** - Graceful queue full behavior

### Feature Pipeline
- SIMD-accelerated microstructure features
- Fibonacci-optimized correlation pruning
- Configurable feature matrix

### Market Regime Detection
- Gaussian Mixture Model classifier
- 3 regime classes (configured in code)
- Real-time probability updates

### Reinforcement Learning
- PPO agent for position sizing
- Shadow fork mode for backtesting
- Drift detection with overseer intervention

### Agentic Overseer
- Self-healing autonomous agent
- Monitors model performance and system health
- Triggers retraining on drift detection

## Troubleshooting

### Import Error: `ImportError: No module named 'antigravity'`
**Solution:** Ensure you've run `pip install -e .` from the project root

### Error: `ModuleNotFoundError: No module named 'winloop'`
This is **expected on Windows**. Uvicorn falls back to default asyncio loop. Performance is only slightly degraded.

### ClickHouse Connection Failed
**Expected behavior** - The server logs a warning and continues without persistence. Data won't be saved between restarts, but all other features work.

### Python version mismatch
**Solution:** Install Python 3.11+ from python.org

### Port 8000 already in use
**Solution:** Kill existing process or change port:
```cmd
python -m uvicorn antigravity.gateway.server:app --host 127.0.0.1 --port 8001
```

## Environment Variables

Key settings (from `antigravity/config.py`):
- `ANTIGRAVITY_GATEWAY_HOST` - Default: "127.0.0.1"
- `ANTIGRAVITY_GATEWAY_PORT` - Default: 8000
- `ANTIGRAVITY_GATEWAY_WS_QUEUE_SIZE` - Default: 100,000
- `ANTIGRAVITY_EXCHANGE_SYMBOLS` - Comma-separated symbols to trade
- `ANTIGRAVITY_CLICKHOUSE_HOST` - ClickHouse server (optional)

See `.env.example` for full configuration template.

## Next Steps

1. **Start the server** using the command in Step 5
2. **Open the dashboard** at http://127.0.0.1:8000/
3. **Monitor logs** for any errors or warnings
4. **Send test WebSocket messages** to `/ws` endpoint
5. **Configure live exchange** by updating credentials via `/api/control/account`

## Key Modules Reference

| Module | Purpose |
|--------|---------|
| `antigravity.gateway.server` | Main FastAPI application entry point |
| `antigravity.features.factory` | Feature computation pipeline |
| `antigravity.regime.classifier` | Market regime detection |
| `antigravity.rl.agent` | Reinforcement learning trader |
| `antigravity.overseer.daemon` | Agentic monitoring system |
| `antigravity.db.client` | ClickHouse persistence layer |

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│       FastAPI Gateway (antigravity.gateway)  │
├─────────────────────────────────────────────┤
│                                              │
│  REST API ──────────── WebSocket (Ticks)   │
│     │                        │               │
│     └────────────────────────┴──────┐       │
│                                      ↓       │
│                          Tick Queue (100K)   │
│                                      │       │
│                                      ↓       │
│           ┌─────────────────────────────┐   │
│           │ Feature Factory             │   │
│           │ (SIMD, Polars)              │   │
│           └──────────────┬──────────────┘   │
│                          ↓                   │
│           ┌─────────────────────────────┐   │
│           │ Regime Classifier           │   │
│           │ (GMM-3 components)          │   │
│           └──────────────┬──────────────┘   │
│                          ↓                   │
│           ┌─────────────────────────────┐   │
│           │ RL Agent (PPO)              │   │
│           │ (Position Sizing)           │   │
│           └──────────────┬──────────────┘   │
│                          ↓                   │
│           ┌─────────────────────────────┐   │
│           │ Agentic Overseer            │   │
│           │ (Drift detection, Healing)  │   │
│           └──────────────┬──────────────┘   │
│                          ↓                   │
│           ┌─────────────────────────────┐   │
│           │ ClickHouse (Persistence)    │   │
│           └─────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

**Created:** Setup Guide for ANTIGRAV TRADING Backend
**Last Updated:** 2024
**Status:** Ready for deployment on localhost
