from stable_baselines3 import PPO
import numpy as np

model_path = "models/ppo_antigrav_latest.zip"
try:
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    
    # Test a few prices in the BTC training range (e.g. 50k to 60k)
    # 0: open, 1: high, 2: low, 3: close, 4: volume, 5: log_return, 6: true_range, 7: norm_atr, 8: last_swing_high, 9: last_swing_low, 10: bullish_bos, 11: bearish_bos, 12: bullish_choch, 13: bearish_choch, 14: padding
    for price in [30000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0]:
        state = np.zeros(15, dtype=np.float32)
        # Set open, high, low, close
        state[0] = price - 50.0
        state[1] = price + 100.0
        state[2] = price - 100.0
        state[3] = price
        state[4] = 5.0
        state[5] = 0.001  # log return
        state[6] = 200.0  # true_range
        state[7] = 0.01   # norm_atr
        state[8] = price + 500.0  # last swing high
        state[9] = price - 500.0  # last swing low
        
        # Test Bullish vs Bearish log returns
        for ret in [-0.01, 0.0, 0.01]:
            state[5] = ret
            action, _ = model.predict(state, deterministic=True)
            print(f"Price: {price:.0f}, Return: {ret:+.2f} -> bias={action[0]:.4f}, kelly={action[1]:.4f}")

except Exception as e:
    print("Error:", e)
