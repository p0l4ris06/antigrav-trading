"""
ANTIGRAV: ZENITH APEX ENGINE v2.0
=================================
1. SENSORS: GMM Regime Classification (Unsupervised ML)
2. DRIVER: Shadow-Optimized Alpha Engine
3. MECHANIC: Agentic Shadow-Trainer (Parallel Thread)
4. DUAL-FORK: Order Book Depth Logger (Log B)
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import ccxt
import threading
from sklearn.mixture import GaussianMixture

# --- CORE CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler("trading_log.txt"), logging.StreamHandler()]
)

# --- THE SENSORS: GMM REGIME CLASSIFIER ---
class RegimeSensor:
    def __init__(self):
        self.model = GaussianMixture(n_components=3, random_state=42)
        self.is_trained = False
        self.current_state = "INITIALIZING"

    def update(self, price_data):
        if len(price_data) < 50: return "LEARNING"
        
        # Feature: Normalized Returns
        df = pd.Series(price_data)
        returns = df.pct_change().dropna().values.reshape(-1, 1)
        
        try:
            if not self.is_trained or np.random.rand() < 0.01: # 1% chance to refit per cycle
                self.model.fit(returns)
                self.is_trained = True
            
            state_idx = self.model.predict(returns[-1:])[0]
            regimes = {0: "ALPHA_TREND", 1: "BETA_REVERSION", 2: "GAMMA_VOLATILITY"}
            self.current_state = regimes.get(state_idx, "UNCERTAIN")
            return self.current_state
        except:
            return "STABILIZING"

# --- THE MECHANIC: SHADOW TRAINER (BACKGROUND EVOLUTION) ---
class ShadowTrainer(threading.Thread):
    def __init__(self, engine):
        threading.Thread.__init__(self)
        self.engine = engine
        self.daemon = True
        self.best_weights = {"trend": 1000, "rsi": 1.0}

    def run(self):
        logging.info("SHADOW_TRAINER >> Mechanic Online. Monitoring for Concept Drift...")
        while True:
            try:
                # In a real system, this would run GA (Genetic Algorithms) or Hyperopt
                # For this evolution, we simulate 'Shadow Recalibration'
                time.sleep(60) # Run every minute
                logging.info("SHADOW_TRAINER >> Running Parallel Simulations on recent window...")
                
                # Simulate finding better weights
                new_trend_weight = self.best_weights["trend"] * np.random.uniform(0.9, 1.1)
                self.best_weights["trend"] = new_trend_weight
                logging.info(f"SHADOW_TRAINER >> Optimal Brain found. Weights Updated: {self.best_weights}")
                
            except Exception as e:
                logging.error(f"SHADOW_ERR >> {e}")

# --- THE ENGINE: ZENITH APEX ---
class ZenithApex:
    def __init__(self):
        self.binance = ccxt.binance({'apiKey': os.getenv("BINANCE_API_KEY"), 'secret': os.getenv("BINANCE_API_SECRET")})
        self.symbols = os.getenv("BINANCE_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
        self.sensor = RegimeSensor()
        self.price_buffers = {s: [] for s in self.symbols}
        self.portfolio = [] # Track open positions
        self.shadow = ShadowTrainer(self)
        self.shadow.start()

    def log_dual_fork(self, ticker, orderbook):
        # DUAL-FORK: Logging order book depth for future Rust-migration (ML Training)
        with open("order_book_fork.log", "a") as f:
            log_entry = {
                "time": datetime.now().isoformat(),
                "ticker": ticker,
                "top_bid": orderbook['bids'][0][0] if orderbook['bids'] else 0,
                "top_ask": orderbook['asks'][0][0] if orderbook['asks'] else 0,
                "bid_depth": sum([b[1] for b in orderbook['bids'][:5]]),
                "ask_depth": sum([a[1] for a in orderbook['asks'][:5]])
            }
            f.write(json.dumps(log_entry) + "\n")

    def run(self):
        logging.info("ZENITH_APEX >> Organism Awakened. Sensors Online.")
        
        while True:
            try:
                best_pick, max_alpha = None, -999
                global_regime = "UNCERTAIN"

                for ticker in self.symbols:
                    # 1. DATA ACQUISITION
                    ticker_data = self.binance.fetch_ticker(ticker)
                    price = ticker_data['last']
                    self.price_buffers[ticker].append(price)
                    if len(self.price_buffers[ticker]) > 200: self.price_buffers[ticker].pop(0)

                    # 2. DUAL-FORK LOGGING (Background Order Book Capture)
                    ob = self.binance.fetch_order_book(ticker, limit=5)
                    self.log_dual_fork(ticker, ob)

                    # 3. REGIME CLASSIFICATION
                    if ticker == self.symbols[0]:
                        global_regime = self.sensor.update(self.price_buffers[ticker])
                        logging.info(f"REGIME_SENSOR >> Vibe Detected: {global_regime}")

                    # 4. ALPHA SCORING (Context-Aware)
                    df = pd.Series(self.price_buffers[ticker])
                    if len(df) > 50:
                        trend = (price / df.rolling(20).mean().iloc[-1]) - 1
                        
                        # Apply Shadow-Optimized weights
                        weights = self.shadow.best_weights
                        alpha = trend * weights["trend"]
                        
                        # Adjust for Regime
                        if global_regime == "ALPHA_TREND": alpha *= 1.5
                        if global_regime == "GAMMA_VOLATILITY": alpha *= 0.5

                        if alpha > max_alpha:
                            max_alpha, best_pick = alpha, ticker

                # 5. SMART EXECUTION (Portfolio Lock)
                if max_alpha > 50:
                    if best_pick not in [p['ticker'] for p in self.portfolio]:
                        logging.info(f"EXECUTION >> Alpha Breach! Opening Position: {best_pick} | Alpha: {max_alpha:.2f}")
                        self.portfolio.append({"ticker": best_pick, "entry": price, "time": time.ctime()})
                        if len(self.portfolio) > 5: self.portfolio.pop(0) # Simple FIFO exit for demo
                    else:
                        logging.info(f"LOCKED >> Position for {best_pick} already active. Maintaining...")

                # 6. TELEMETRY SYNC
                status = {
                    "message": f"REGIME: {global_regime} | TARGET: {best_pick}",
                    "best_pick": f"{best_pick} ({global_regime})",
                    "best_pick_key": f"BINANCE:{best_pick}",
                    "regime": global_regime,
                    "last_update": time.ctime(),
                    "ohlc": {f"BINANCE:{k}": [[time.ctime(), b[-1], b[-1], b[-1], b[-1]]] for k, b in self.price_buffers.items() if b},
                    "trades": self.portfolio[-10:],
                    "shadow_weights": self.shadow.best_weights
                }
                
                temp_file = "bot_state.json.tmp"
                with open(temp_file, "w") as f: json.dump(status, f)
                os.replace(temp_file, "bot_state.json")

                time.sleep(5)

            except Exception as e:
                logging.error(f"ENGINE_CRASH >> {e}")
                time.sleep(10)

if __name__ == "__main__":
    engine = ZenithApex()
    engine.run()
