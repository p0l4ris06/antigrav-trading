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
    
    # 1. Base vector: realistic BTC-like values
    base_price = 55000.0
    base_state = np.zeros(15, dtype=np.float32)
    base_state[0] = base_price - 50.0  # open
    base_state[1] = base_price + 100.0 # high
    base_state[2] = base_price - 100.0 # low
    base_state[3] = base_price         # close
    base_state[4] = 5.0                # volume
    base_state[5] = 0.0                # log_return
    base_state[6] = 200.0              # true_range
    base_state[7] = 0.01               # norm_atr
    base_state[8] = base_price + 500.0 # last_swing_high
    base_state[9] = base_price - 500.0 # last_swing_low
    
    # Check default prediction
    default_act, _ = model.predict(base_state, deterministic=True)
    print(f"Base State Predict -> bias={default_act[0]:.4f}, kelly={default_act[1]:.4f}\n")
    
    # 2. Perturb each feature individually
    for i, name in enumerate(feature_names):
        print(f"--- Perturbing '{name}' (Index {i}) ---")
        # Find appropriate test range for this feature
        if i in [0, 1, 2, 3, 8, 9]:  # Price features
            test_vals = [0.0, 1000.0, 10000.0, 30000.0, 50000.0, 70000.0, 100000.0]
        elif i == 4:  # Volume
            test_vals = [0.0, 1.0, 10.0, 100.0, 1000.0]
        elif i == 5:  # Log return
            test_vals = [-0.1, -0.05, -0.01, -0.001, 0.0, 0.001, 0.01, 0.05, 0.1]
        elif i in [6, 7]:  # Volatility/ATR
            test_vals = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        else:  # BOS/CHOCH/Padding flags
            test_vals = [0.0, 0.5, 1.0, 2.0]
            
        for val in test_vals:
            state = base_state.copy()
            state[i] = val
            action, _ = model.predict(state, deterministic=True)
            if np.sign(action[0]) != np.sign(default_act[0]):
                print(f"  * VALUE SHIFT * {name}={val} -> bias={action[0]:+.4f}, kelly={action[1]:.4f}")
            else:
                pass # print(f"  {name}={val} -> bias={action[0]:+.4f}")
                
except Exception as e:
    print("Error:", e)
