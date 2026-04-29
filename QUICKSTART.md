# ANTIGRAV TRADING - Quick Start (30 seconds)

## Prerequisites
- Python 3.11+ installed
- You're in: `C:\Users\Wren C\Documents\ANTIGRAV TRADING\`

## One-Time Setup
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -e .
```

## Start Backend Server
```cmd
python -m uvicorn antigravity.gateway.server:app --host 127.0.0.1 --port 8000 --reload
```

## What You'll See
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     gateway.starting
INFO:     gateway.feature_factory_ready active_features=[...]
INFO:     Application startup complete
```

## Connect to Server
- **Dashboard**: http://127.0.0.1:8000/
- **Status API**: http://127.0.0.1:8000/api/status
- **WebSocket**: ws://127.0.0.1:8000/ws

## Stop Server
Press `Ctrl+C` in the terminal

## If It Fails
1. Check Python version: `python --version`
2. Check venv activated: You should see `(venv)` in your prompt
3. Check pip packages: `pip list`
4. Full guide: See `BACKEND_SETUP.md`

---
**Entry Point:** `antigravity.gateway.server:app`
**Database:** ClickHouse (optional, graceful fallback)
**Features:** Auto-simulation enabled by default
