from stable_baselines3 import PPO
import numpy as np

model_path = "models/ppo_antigrav_latest.zip"
try:
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    
    # 1. Base State (all zeros)
    state_zeros = np.zeros(15, dtype=np.float32)
    action, _ = model.predict(state_zeros, deterministic=True)
    print(f"Zeros input -> bias={action[0]:.4f}, kelly={action[1]:.4f}")
    
    # Let's test different log returns (index 5)
    # Bullish return: +5% (0.05)
    state_bullish = np.zeros(15, dtype=np.float32)
    state_bullish[3] = 60000.0  # close price
    state_bullish[5] = 0.05     # log return
    state_bullish[11] = 1.0     # bullish_bos
    action, _ = model.predict(state_bullish, deterministic=True)
    print(f"Bullish input -> bias={action[0]:.4f}, kelly={action[1]:.4f}")
    
    # Bearish return: -5% (-0.05)
    state_bearish = np.zeros(15, dtype=np.float32)
    state_bearish[3] = 60000.0  # close price
    state_bearish[5] = -0.05    # log return
    state_bearish[12] = 1.0     # bearish_bos
    action, _ = model.predict(state_bearish, deterministic=True)
    print(f"Bearish input -> bias={action[0]:.4f}, kelly={action[1]:.4f}")

    # Test random returns
    for ret in [-0.1, -0.01, 0.0, 0.01, 0.1]:
        state = np.zeros(15, dtype=np.float32)
        state[3] = 60000.0
        state[5] = ret
        action, _ = model.predict(state, deterministic=True)
        print(f"Log Return {ret:+.2f} -> bias={action[0]:.4f}, kelly={action[1]:.4f}")

except Exception as e:
    print("Error during model diagnosis:", e)
