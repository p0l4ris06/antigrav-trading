# Changelog

All notable changes to the **ANTIGRAV TRADING** quantitative platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-07-30

### Added
- **3,000 ms minimum position age (`MIN_REVERSAL_AGE_MS = 3000`)**: `PaperPosition` now records `opened_timestamp_ms`; opposing orders within the minimum age are blocked (`reversal.blocked_too_young`).
- **Server & Client Persistent Micro-Trends**: Integrated 4-12 second persistent symbol micro-trend drift into `ws_simulated_feed` and `useMarketData.tsx`.
- **Top 25 Liquid Asset Filter**: Restricted Alpaca dynamic asset listings to top 25 liquid instruments with category tabs and live search bar.
- **Structured Audit Logging**: Added CSV (`data/paper_trade_log.csv`) and JSON lines (`data/paper_trades_audit.jsonl`) automatic trade logging.
- **Pytest Suite (`tests/test_paper_engine.py`)**: Unit tests covering position timing, reversal protection, and PnL calculations.
- **GitHub Actions CI Pipeline (`.github/workflows/ci.yml`)**: Automated testing, ruff linting, and dashboard build validation.
- **Containerization**: Added `Dockerfile` and `docker-compose.yml` for unified backend & dashboard deployment.

### Fixed
- Fixed single-symbol websocket loop bottleneck in `server.py` to stream multi-asset ticks concurrently across all symbols.
- Resolved target profit stop re-trigger glitch when setting custom balance or resuming auto-trade.
- Fixed 100ms signal flipping caused by randomized per-frame tick directions.

---

## [1.1.0] - 2026-07-29

### Added
- Real-time RL Autoresearch Experience Reloop telemetry streaming via `/api/status`.
- Persistence of historical paper trades in `data/paper_experience_replay.jsonl`.
- Overseer rolling Sharpe ratio baseline correction (`+1.842`).
