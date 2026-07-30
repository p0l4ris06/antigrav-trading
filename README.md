# 🚀 ANTIGRAV TRADING

> Autonomous RL-driven quantitative trading system with self-healing agentic overseer and live execution engine.

---

## 📌 Overview

**Antigrav Trading** is a high-performance, modular algorithmic trading platform built in Python, TypeScript, and Rust. It combines reinforcement learning (PPO with Kelly-Convexity optimization), real-time order book imbalance (OBI) features, unsupervised GMM regime classification, and an agentic self-healing overseer daemon.

---

## ✨ Key Features

- **RL Trading Engine**: Gym-compatible `KellyConvexEnv` trained via PPO (`stable-baselines3`) with walk-forward evaluation.
- **Regime Classification**: GMM & PCA unsupervised clustering for detecting market state shifts (Trending, Ranging, High Volatility).
- **Agentic Overseer**: Self-healing daemon tracking Sharpe drift, drawdowns, and model performance with automated refitting.
- **FastAPI & WebSockets Gateway**: High-throughput REST API and real-time WebSocket feeds for price and telemetry broadcasting.
- **React Dashboard**: Modern Vite + React + Tailwind frontend featuring live trading execution, Order Book Imbalance (OBI) visualization, and overseer event logs.
- **Multi-Exchange Adapters**: Support for Alpaca (Stocks & Crypto) and CCXT (Binance, Bybit, Coinbase, Trading212).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Live Market Data & Feeds                    │
│             (Alpaca / CCXT / Trading212)                │
└──────────────────────────┬──────────────────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │  Feature Factory  │
                 │  (15m/4H + OBI)   │
                 └─────────┬─────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│ PPO Agent   │     │  Regime     │     │ Agentic     │
│ (RL Engine) │     │ Classifier  │     │ Overseer    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │ FastAPI Gateway   │
                 │  (REST + WebSockets)
                 └─────────┬─────────┘
                           │
                 ┌─────────▼─────────┐
                 │ React Dashboard   │
                 │ (Live Telemetry)  │
                 └───────────────────┘
```

---

## 🛠️ Quick Start

### 1. Repository Setup & Dependencies

```bash
# Clone the repository
git clone https://github.com/p0l4ris06/antigrav-trading.git
cd antigrav-trading

# Create virtual environment and install backend
uv venv
uv pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and populate your API credentials:

```bash
cp .env.example .env
```

### 3. Launch Backend Gateway

```bash
uv run antigrav serve
```
*The FastAPI backend will start at `http://localhost:8000`.*

### 4. Launch Frontend Dashboard

```bash
cd dashboard
npm install
npm run dev
```
*The dashboard will be available at `http://localhost:5173`.*

---

## 📂 Repository Structure

```
ANTIGRAV TRADING/
├── src/antigravity/              ← Core backend infrastructure layer
│   ├── gateway/                  ← FastAPI server & WebSocket consumers
│   ├── overseer/                 ← Agentic self-healing daemon
│   ├── regime/                   ← Unsupervised PCA + GMM regime classifier
│   ├── rl/                       ← Gymnasium environment & PPO training
│   ├── exchange/                 ← Alpaca & CCXT multi-exchange adapters
│   └── telemetry/                ← OpenTelemetry & LangSmith tracing
├── src/cli/                      ← CLI commands (antigrav serve, live, backtest)
├── dashboard/                    ← React + Vite + Tailwind dashboard
├── docs/                         ← System architecture and API guides
├── scripts/                      ← Utility setup and maintenance scripts
├── pyproject.toml                ← Package configuration & dependencies
└── .env.example                  ← Environment configuration template
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
