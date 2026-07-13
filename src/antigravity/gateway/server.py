"""
Antigravity Gateway Server.

High-throughput FastAPI application backed by winloop (Windows) or uvloop (Linux).
Handles:
    - WebSocket tick ingestion with asyncio.Queue backpressure
    - REST control plane for system status and manual overrides
    - Lifecycle management for ClickHouse, Feature Factory, Regime, RL, Overseer
    - Simulated + live exchange adapters
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from antigravity.config import settings
from antigravity.storage.clickhouse import ClickHouseManager
from antigravity.features.factory import FeatureFactory
from antigravity.gateway.websocket.tick_consumer import TickConsumer
from antigravity.overseer.daemon import AgenticOverseer
from antigravity.regime.classifier import RegimeClassifier
from antigravity.telemetry.tracing import init_tracing, shutdown_tracing
from antigravity.rl.backtester import AntigravBacktester

logger = structlog.get_logger(__name__)


async def _auto_simulate(queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Background task to generate synthetic ticks for all configured symbols."""
    import random
    
    symbols = settings.exchange.symbols
    prices = {s: 50000.0 if "BTC" in s else 3000.0 if "ETH" in s else 100.0 for s in symbols}
    
    logger.info("sim.auto_start", symbols=symbols)
    
    while True:
        for symbol in symbols:
            # Random walk
            prices[symbol] += random.gauss(0, prices[symbol] * 0.0001)
            spread = prices[symbol] * 0.0002
            
            tick = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bid_price": round(prices[symbol] - spread/2, 4),
                "ask_price": round(prices[symbol] + spread/2, 4),
                "last_price": round(prices[symbol], 4),
                "last_size": round(random.expovariate(1.0), 4),
            }
            
            try:
                queue.put_nowait(tick)
            except asyncio.QueueFull:
                pass
                
        await asyncio.sleep(0.5) # 2 ticks per second per symbol

# ---------------------------------------------------------------------------
# Install high-performance event loop
# ---------------------------------------------------------------------------
def _install_event_loop() -> None:
    """Install winloop on Windows, uvloop on Linux/macOS."""
    if sys.platform == "win32":
        try:
            import winloop  # type: ignore[import-untyped]

            winloop.install()
            logger.info("event_loop.winloop_installed")
        except ImportError:
            logger.warning("event_loop.winloop_unavailable, falling back to default")
    else:
        try:
            import uvloop  # type: ignore[import-untyped]

            uvloop.install()
            logger.info("event_loop.uvloop_installed")
        except ImportError:
            logger.warning("event_loop.uvloop_unavailable, falling back to default")


# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------
class TickData(BaseModel):
    """Incoming tick data from exchange WebSocket."""

    symbol: str = "BTCUSDT"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bid_price: float
    ask_price: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: float = 0.0
    last_size: float = 0.0
    trade_id: int = 0


class SystemStatus(BaseModel):
    """System health snapshot."""

    uptime_seconds: float = 0.0
    ticks_ingested: int = 0
    buffer_height: int = 0
    current_regime: str = "unknown"
    regime_probabilities: list[float] = []
    portfolio_weights: list[float] = []
    rolling_sharpe: float = 0.0
    overseer_state: str = "MONITORING"
    overseer_events: list[dict[str, Any]] = []
    drift_detected: bool = False
    shadow_fork_active: bool = False
    last_price: float = 0.0
    prices: dict[str, float] = {}
    queue_depth: int = 0
    active_features: list[str] = []
    regime_n_components: int = 0
    ppo_model_loaded: bool = False


class ControlAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    FORCE_RETRAIN = "force_retrain"
    FORCE_REFIT_REGIME = "force_refit_regime"
    PRUNE_FEATURES = "prune_features"
    RESET_FEATURES = "reset_features"


# ---------------------------------------------------------------------------
# Application State (shared across request handlers)
# ---------------------------------------------------------------------------
class ExecutionUpdate(BaseModel):
    enabled: bool
    max_position_size: float


class AccountUpdate(BaseModel):
    api_key: str
    api_secret: str


class AppState:
    """Mutable application state container."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.ticks_ingested: int = 0
        self.tick_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=settings.gateway.ws_queue_size
        )
        self.ch_manager: ClickHouseManager | None = None
        self.consumer: TickConsumer | None = None
        self.feature_factory: FeatureFactory | None = None
        self.regime_classifier: RegimeClassifier | None = None
        self.agent: Any = None  # AgentManager (lazy-loaded if model exists)
        self.overseer: AgenticOverseer | None = None
        self.paused: bool = False
        self.current_regime: str = "unknown"
        self.regime_probabilities: list[float] = []
        self.portfolio_weights: list[float] = []
        self.rolling_sharpe: float = 0.0
        self.overseer_state: str = "MONITORING"
        self.drift_detected: bool = False
        self.shadow_fork_active: bool = False
        self.last_price: float = 0.0
        self.prices: dict[str, float] = {}
        self.simulation_task: asyncio.Task | None = None
        self.ppo_model: Any = None
        self.execution_enabled: bool = True


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown hooks."""
    logger.info("gateway.starting")

    # --- OpenTelemetry / LangSmith ---
    # Must init before creating the FastAPI app middleware
    # We pass `app` later after yield for auto-instrumentation
    _tracing_ok = init_tracing(app)

    # --- ClickHouse ---
    try:
        app_state.ch_manager = await ClickHouseManager.create()
        await app_state.ch_manager.ensure_schema()
        logger.info("gateway.clickhouse_ready")
    except Exception as exc:
        logger.error("gateway.clickhouse_failed", error=str(exc))
        logger.warning("gateway.running_without_persistence")

    # --- Load trained PPO model if available ---
    models_dir = Path(os.getenv("AG_MODEL_DIR", "models"))
    ppo_model_path = models_dir / "ppo_antigrav_latest.zip"
    if ppo_model_path.exists():
        try:
            from stable_baselines3 import PPO
            app_state.ppo_model = PPO.load(str(ppo_model_path))
            logger.info("gateway.ppo_model_loaded", path=str(ppo_model_path),
                        obs_shape=app_state.ppo_model.observation_space.shape)
        except Exception as exc:
            logger.warning("gateway.ppo_model_load_failed", error=str(exc))
            app_state.ppo_model = None
    else:
        logger.warning("gateway.no_ppo_model", hint="Run train.py first")
        app_state.ppo_model = None

    # --- Feature Factory ---
    app_state.feature_factory = FeatureFactory()
    
    # Dynamically prune correlated features to prevent dimensionality curse
    # Uses Spearman rank correlation threshold (default 0.85)
    if app_state.ch_manager:
        # Load recent data for pruning decision
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=24)
        try:
            df = await app_state.ch_manager.fetch_data(symbol="BTCUSDT", start=start, end=now)
            if df is not None and not df.is_empty():
                dropped = app_state.feature_factory.prune_correlated_features(df)
                logger.info("gateway.features_pruned", dropped=dropped)
        except Exception:
            pass
            
    logger.info("gateway.feature_factory_ready", active_features=app_state.feature_factory.get_feature_names())

    if app_state.feature_factory and app_state.ppo_model:
        expected_dim = app_state.ppo_model.observation_space.shape[0]
        actual_features = len(app_state.feature_factory.get_feature_names())
        if actual_features == 0:
            actual_features = 15  # Fallback to canonical features size if not computed yet
        if actual_features != expected_dim:
            logger.error(
                "Feature/model mismatch: model expects %d dims, factory produces %d",
                expected_dim, actual_features
            )

    # --- Regime Classifier ---
    app_state.regime_classifier = RegimeClassifier()
    regime_path = Path("models") / "regime_classifier.pkl"
    n_regimes = 3  # default
    if regime_path.exists():
        try:
            import pickle
            with open(regime_path, "rb") as f:
                app_state.regime_classifier = pickle.load(f)
            n_regimes = app_state.regime_classifier.n_components
            logger.info("gateway.regime_classifier_ready", fitted=True, n_regimes=n_regimes)
        except Exception as exc:
            logger.warning("gateway.regime_classifier_load_failed", error=str(exc))
    else:
        logger.info("gateway.regime_classifier_ready", fitted=False)

    # --- RL Agent (load if pre-trained model exists) ---
    model_path = Path(os.getenv("AG_MODEL_DIR", "models"))
    if (model_path / "ppo_model.zip").exists():
        try:
            import numpy as np
            from antigravity.rl.agent import AgentManager
            from antigravity.rl.environment import TradingEnv

            # Determine feature count (from factory or model metadata)
            # For now, we align with the 8 features used in training + n_regimes
            n_features = 8 
            obs_dim = n_features + n_regimes

            # Create a dummy env with correct dimensions
            dummy_env = TradingEnv(
                feature_data=np.zeros((100, n_features), dtype=np.float32),
                price_data=np.ones(100, dtype=np.float64) * 50000,
                atr_data=np.ones(100, dtype=np.float64),
                regime_data=np.zeros((100, n_regimes), dtype=np.float32),
            )
            app_state.agent = AgentManager(env=dummy_env)
            app_state.agent.load(model_path)
            logger.info("gateway.agent_loaded", path=str(model_path), obs_dim=obs_dim)
        except Exception as exc:
            logger.warning("gateway.agent_load_failed", error=str(exc))
    else:
        logger.info("gateway.no_pretrained_model", hint="Run `python -m antigravity.train` first")

    # --- Overseer ---
    app_state.overseer = AgenticOverseer(
        agent=app_state.agent,
        feature_factory=app_state.feature_factory,
        regime_classifier=app_state.regime_classifier,
        ch_manager=app_state.ch_manager,
        app_state=app_state,
    )
    overseer_task = asyncio.create_task(app_state.overseer.run())
    logger.info("gateway.overseer_started")

    # --- Tick Consumer (wired to all components) ---
    app_state.consumer = TickConsumer(
        queue=app_state.tick_queue,
        ch_manager=app_state.ch_manager,
        app_state=app_state,
        feature_factory=app_state.feature_factory,
        regime_classifier=app_state.regime_classifier,
        agent=app_state.agent,
        overseer=app_state.overseer,
    )
    consumer_task = asyncio.create_task(app_state.consumer.run())

    # --- Live Exchange Adapter (optional) ---
    exchange_task = None
    exchange_adapter = None
    if settings.exchange.enabled:
        try:
            from antigravity.exchange.binance import BinanceAdapter, BinanceMultiAdapter

            symbols = settings.exchange.symbols
            if len(symbols) == 1:
                exchange_adapter = BinanceAdapter(
                    symbol=symbols[0],
                    queue=app_state.tick_queue,
                    futures=settings.exchange.futures,
                )
            else:
                exchange_adapter = BinanceMultiAdapter(
                    symbols=symbols,
                    queue=app_state.tick_queue,
                    futures=settings.exchange.futures,
                )
            exchange_task = asyncio.create_task(exchange_adapter.run())
            logger.info(
                "gateway.exchange_adapter_started",
                adapter="binance",
                symbols=symbols,
                futures=settings.exchange.futures,
            )
        except Exception as exc:
            logger.error("gateway.exchange_adapter_failed", error=str(exc))
    else:
        logger.info(
            "gateway.exchange_adapter_disabled",
            hint="Set AG_EXCHANGE_ENABLED=true and connect to /ws/simulated_feed for dev",
        )

    logger.info("gateway.ready", port=settings.gateway.port)

    # --- Auto-Simulation (Background) ---
    if not settings.exchange.enabled:
        app_state.simulation_task = asyncio.create_task(_auto_simulate(app_state.tick_queue))
        logger.info("gateway.auto_simulation_started")

    yield

    # Shutdown
    logger.info("gateway.shutting_down")
    app_state.consumer.stop()
    consumer_task.cancel()
    overseer_task.cancel()

    if exchange_adapter:
        exchange_adapter.stop()
    if exchange_task:
        exchange_task.cancel()
        try:
            await exchange_task
        except asyncio.CancelledError:
            pass

    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    try:
        await overseer_task
    except asyncio.CancelledError:
        pass

    if app_state.simulation_task:
        app_state.simulation_task.cancel()
    
    if app_state.ch_manager:
        await app_state.ch_manager.close()

    # Flush pending OTel spans
    shutdown_tracing()

    logger.info("gateway.shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Antigravity Trading Engine",
    description="Autonomous RL-driven quantitative trading system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket Endpoint ÔÇö Tick Ingestion
# ---------------------------------------------------------------------------
@app.websocket("/ws/ingest")
async def ws_tick_ingest(websocket: WebSocket) -> None:
    """
    Accepts JSON tick data from exchange adapters.

    Tick format (JSON):
        {
            "symbol": "BTCUSDT",
            "bid_price": 50000.5,
            "ask_price": 50001.0,
            "bid_size": 1.2,
            "ask_size": 0.8,
            "last_price": 50000.75,
            "last_size": 0.5,
            "trade_id": 12345
        }

    Pushes to bounded asyncio.Queue for downstream processing.
    Backpressure: if queue is full, tick is dropped with a warning.
    """
    await websocket.accept()
    peer = websocket.client
    logger.info("ws.connected", peer=str(peer))

    try:
        while True:
            if app_state.paused:
                await asyncio.sleep(0.1)
                continue

            raw = await websocket.receive_json()

            # Validate and hydrate defaults
            tick = TickData.model_validate(raw)
            tick_dict = tick.model_dump()

            # Non-blocking enqueue with backpressure
            try:
                app_state.tick_queue.put_nowait(tick_dict)
                app_state.ticks_ingested += 1
                app_state.last_price = tick_dict.get("last_price", 0.0)
            except asyncio.QueueFull:
                try:
                    await websocket.send_json({
                        "type": "backpressure",
                        "queue_depth": app_state.tick_queue.qsize(),
                        "message": "Please slow down"
                    })
                except Exception:
                    pass
                logger.warning("ws.backpressure", queue_size=app_state.tick_queue.qsize())

    except WebSocketDisconnect:
        logger.info("ws.disconnected", peer=str(peer))
    except Exception as exc:
        logger.error("ws.error", error=str(exc), peer=str(peer))


# ---------------------------------------------------------------------------
# WebSocket Endpoint ÔÇö Simulated Exchange Feed (for development)
# ---------------------------------------------------------------------------
@app.websocket("/ws/simulated_feed")
async def ws_simulated_feed(websocket: WebSocket) -> None:
    """
    Sends simulated L2 order book ticks for development/testing.
    Connect any WebSocket client to receive a realistic tick stream.
    """
    import random

    await websocket.accept()
    logger.info("ws.simulated_feed.started")

    base_price = 50_000.0
    tick_id = 0

    try:
        while True:
            # Random walk with mean reversion
            base_price += random.gauss(0, 5.0)
            spread = abs(random.gauss(0.5, 0.2))
            bid = base_price - spread / 2
            ask = base_price + spread / 2

            tick = {
                "symbol": "BTCUSDT",
                "bid_price": round(bid, 2),
                "ask_price": round(ask, 2),
                "bid_size": round(random.expovariate(1 / 2.0), 4),
                "ask_size": round(random.expovariate(1 / 2.0), 4),
                "last_price": round(bid + random.random() * spread, 2),
                "last_size": round(random.expovariate(1 / 0.5), 4),
                "trade_id": tick_id,
            }
            tick_id += 1

            await websocket.send_json(tick)

            # Also push to ingestion queue for self-feeding
            try:
                tick["timestamp"] = datetime.now(timezone.utc).isoformat()
                app_state.tick_queue.put_nowait(tick)
                app_state.ticks_ingested += 1
                app_state.last_price = tick.get("last_price", 0.0)
            except asyncio.QueueFull:
                pass

            # ~10 ticks/second for simulation
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("ws.simulated_feed.stopped")
    except Exception as exc:
        logger.error("ws.simulated_feed.error", error=str(exc))


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def get_status():
    """Return current system health snapshot."""
    return {
        "current_regime": app_state.current_regime,
        "overseer_state": app_state.overseer_state,
        "rolling_sharpe": round(app_state.rolling_sharpe, 4),
        "drift_detected": app_state.drift_detected,
        "shadow_fork_active": app_state.shadow_fork_active,
        "ticks_ingested": app_state.ticks_ingested,
        "buffer_height": getattr(app_state.feature_factory, "buffer_height", 0)
                         if app_state.feature_factory else 0,
        "uptime_seconds": int(time.time() - app_state.start_time),
        "overseer_events": app_state.overseer.event_log if app_state.overseer else [],
        "regime_probabilities": app_state.regime_probabilities,
        "portfolio_weights": app_state.portfolio_weights,
        "ppo_model_loaded": app_state.ppo_model is not None,
        "prices": app_state.prices,
    }


@app.get("/api/backtest/run")
async def run_backtest(symbol: str = "BTC"):
    """Trigger an offline backtest and return performance metrics."""
    backtester = AntigravBacktester(symbol=symbol)
    backtester.load_model()
    
    # Use data from the data/ directory
    data_path = f"data/{symbol}_USDT_15m.parquet"
    if not os.path.exists(data_path):
        # Fallback to whatever parquet is available
        data_files = [f for f in os.listdir("data") if f.endswith(".parquet")]
        if not data_files:
            return {"status": "failed", "detail": "No historical data found"}
        data_path = os.path.join("data", data_files[0])

    # Run in thread to avoid blocking the gateway's event loop
    equity_curve, actions = await asyncio.to_thread(backtester.run, data_path)
    
    # Calculate summary metrics
    final_return = (equity_curve[-1] - 1.0) * 100
    return {
        "status": "success",
        "symbol": symbol,
        "data_path": data_path,
        "final_return_pct": round(final_return, 2),
        "final_equity": round(equity_curve[-1], 4),
        "equity_curve": equity_curve[::10],  # Downsample for JSON transport
    }


@app.post("/api/control/{action}")
async def control_action(action: ControlAction) -> dict[str, str]:
    """Manual system overrides."""
    match action:
        case ControlAction.PAUSE:
            app_state.paused = True
            return {"status": "paused", "detail": "Tick ingestion paused"}
        case ControlAction.RESUME:
            app_state.paused = False
            return {"status": "resumed", "detail": "Tick ingestion resumed"}
        case ControlAction.FORCE_RETRAIN:
            app_state.overseer_state = "FORCE_RETRAIN_REQUESTED"
            return {"status": "accepted", "detail": "Force retrain signal sent to overseer"}
        case ControlAction.FORCE_REFIT_REGIME:
            if app_state.feature_factory and app_state.regime_classifier:
                df = app_state.feature_factory.compute_features()
                if df is not None:
                    feature_cols = app_state.feature_factory.get_feature_names()
                    available = [c for c in feature_cols if c in df.columns]
                    if available:
                        matrix = df.select(available).drop_nulls().to_numpy()
                        stats = app_state.regime_classifier.fit(matrix)
                        return {"status": "refitted", "detail": str(stats)}
            return {"status": "failed", "detail": "Insufficient data for regime refit"}
        case ControlAction.PRUNE_FEATURES:
            if app_state.feature_factory:
                dropped = app_state.feature_factory.prune_correlated_features()
                return {"status": "pruned", "detail": f"Dropped: {dropped}"}
            return {"status": "failed", "detail": "Feature factory not initialized"}
        case ControlAction.RESET_FEATURES:
            if app_state.feature_factory:
                app_state.feature_factory.reset_feature_selection()
                return {"status": "reset", "detail": "All features re-enabled"}
            return {"status": "failed", "detail": "Feature factory not initialized"}


@app.post("/api/control/action")
async def control_action_v2(payload: dict[str, Any] = {}):
    """Handle dashboard action buttons (retrain, kill switch)."""
    action = payload.get("action", "")

    if action == "retrain":
        import subprocess
        import threading

        def _run_retrain():
            subprocess.run([
                "python", "train.py",
                "--data", "data/BTC_USDT_15m.parquet",
                          "data/ETH_USDT_15m.parquet",
                          "data/SOL_USDT_15m.parquet",
                "--timesteps", "50000"
            ])
            # Reload model after training
            from stable_baselines3 import PPO
            try:
                model_zip_path = Path(os.getenv("AG_MODEL_DIR", "models")) / "ppo_antigrav_latest.zip"
                app_state.ppo_model = PPO.load(str(model_zip_path))
                logger.info("gateway.ppo_model_reloaded_after_retrain")
            except Exception as exc:
                logger.error("gateway.retrain_reload_failed", error=str(exc))

        threading.Thread(target=_run_retrain, daemon=True).start()
        app_state.overseer_state = "RETRAINING"
        return {"success": True, "message": "Retraining started in background"}

    elif action == "kill":
        logger.warning("gateway.kill_switch_activated")
        app_state.execution_enabled = False
        app_state.overseer_state = "KILLED"
        # Write a halt file that live_daemon.py checks
        Path(".daemon_halt").touch()
        return {"success": True, "message": "Kill switch activated. Live daemon will halt."}

    return {"success": False, "message": f"Unknown action: {action}"}


@app.get("/api/research/status")
async def research_status():
    """Returns the latest autoresearcher experiment history."""
    import json as _json
    history_path = Path("experiments/history.jsonl")
    if not history_path.exists():
        return {"entries": [], "best_score": None}
    entries = []
    best = None
    with open(history_path) as f:
        for line in f:
            try:
                rec = _json.loads(line)
                entries.append(rec)
                s = rec.get("score")
                if s and (best is None or s > best):
                    best = s
            except Exception:
                continue
    return {
        "entries": entries[-50:],
        "best_score": best,
        "total_iterations": len(entries),
    }


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/overseer/events")
async def get_overseer_events(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent overseer events for the dashboard."""
    if app_state.overseer:
        return app_state.overseer.event_log[-limit:]
    return []


@app.get("/api/features")
async def get_feature_info() -> dict[str, Any]:
    """Return current feature factory state."""
    if not app_state.feature_factory:
        return {"error": "not_initialized"}
    ff = app_state.feature_factory
    return {
        "buffer_height": ff.buffer_height,
        "active_features": ff.get_feature_names(),
        "atr": ff.get_current_atr(),
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the gateway server."""
    _install_event_loop()

    uvicorn.run(
        "antigravity.gateway.server:app",
        host=settings.gateway.host,
        port=settings.gateway.port,
        log_level="info",
        reload=False,
    )


@app.post("/api/control/execution")
async def update_execution(update: ExecutionUpdate) -> dict[str, str]:
    """Enable/disable auto-trading."""
    settings.execution.enabled = update.enabled
    settings.execution.max_position_size = update.max_position_size
    logger.info("execution.updated", enabled=update.enabled)
    return {"status": "success"}


@app.post("/api/control/account")
async def update_account(update: AccountUpdate) -> dict[str, str]:
    """Update exchange credentials."""
    settings.account.api_key = update.api_key
    settings.account.api_secret = update.api_secret
    logger.info("account.credentials_updated")
    return {"status": "success"}


# ---------------------------------------------------------------------------
# Static Files (Dashboard)
# ---------------------------------------------------------------------------
from fastapi.responses import FileResponse

# Resolve dashboard path relative to this file's directory
# (server.py is in antigravity/gateway/, so root is ../../dashboard/dist)
base_dir = Path(__file__).parent.parent.parent
dashboard_dist = base_dir / "dashboard" / "dist"

@app.get("/")
async def serve_dashboard():
    """Serve the dashboard index.html."""
    index_path = dashboard_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Dashboard not built. Run `npm run build` in /dashboard"}

if dashboard_dist.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dist), html=True), name="dashboard")
    logger.info("gateway.dashboard_mounted", path=str(dashboard_dist))
else:
    logger.warning("gateway.dashboard_not_found", path=str(dashboard_dist), hint="Run `npm run build` in dashboard directory")


if __name__ == "__main__":
    main()
