from stable_baselines3 import PPO
import numpy as np

model_path = "models/ppo_antigrav_latest.zip"
try:
    model = PPO.load(model_path)
    print("Model loaded.")
    
    # Feature names in order
    feature_names = [
        "open", "high", "low", "close", "volume", "log_return",
        "true_range", "norm_atr", "last_swing_high", "last_swing_low",
        "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch", "padding"
    ]
    
    # Start with mostly-zero state that gives LONG
    state = np.zeros(15, dtype=np.float32)
    state[3] = 60000.0  # close price
    state[5] = 0.05     # log return
    state[11] = 1.0     # bearish_bos
    
    action, _ = model.predict(state, deterministic=True)
    print(f"Step 0 (Bullish State) -> bias={action[0]:+.4f}, kelly={action[1]:.4f}\n")
    
    # We will replace each element with its realistic value one-by-one
    realistic_vals = {
        "open": 54950.0,
        "high": 55100.0,
        "low": 54900.0,
        "volume": 5.0,
        "true_range": 200.0,
        "norm_atr": 0.01,
        "last_swing_high": 55500.0,
        "last_swing_low": 54500.0,
    }
    
    for name, val in realistic_vals.items():
        idx = feature_names.index(name)
        state[idx] = val
        action, _ = model.predict(state, deterministic=True)
        print(f"Set '{name}' (Index {idx}) to {val} -> bias={action[0]:+.4f}, kelly={action[1]:.4f}")
        
except Exception as e:
    print("Error:", e)
