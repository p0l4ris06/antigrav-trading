from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import pickle
import numpy as np

model_path = "models/ppo_model.zip"
vec_norm_path = "models/vec_normalize.pkl"

try:
    model = PPO.load(model_path)
    print("Large PPO model loaded successfully.")
    print("Observation space shape:", model.observation_space.shape)
    
    with open(vec_norm_path, "rb") as f:
        vec_norm = pickle.load(f)
    print("VecNormalize loaded successfully.")
    print("Normalizer mean shape:", vec_norm.obs_rms.mean.shape)
    
    # Let's generate a realistic feature vector for the normalizer
    # The normalizer expects 14 features
    raw_obs_14 = np.random.randn(14).astype(np.float32)
    
    # Normalize and pad/truncate to match model observation space (which is size 14 or 15 or 17?)
    norm_obs = vec_norm.normalize_obs(raw_obs_14)
    print("Normalized 14:", norm_obs)
    
    expected_dim = model.observation_space.shape[0]
    print("Model expects observation dimension:", expected_dim)
    
    # Pad or truncate to expected_dim
    if len(norm_obs) < expected_dim:
        state = np.pad(norm_obs, (0, expected_dim - len(norm_obs)), mode='constant')
    else:
        state = norm_obs[:expected_dim]
        
    action, _ = model.predict(state, deterministic=True)
    print(f"Action prediction -> bias={action[0]:.4f}")
    
    # Let's test a batch of different features
    print("\nTesting responsiveness with different raw features:")
    for i in range(5):
        raw_obs = np.random.randn(14).astype(np.float32) * (i + 1)
        norm_obs = vec_norm.normalize_obs(raw_obs)
        if len(norm_obs) < expected_dim:
            state = np.pad(norm_obs, (0, expected_dim - len(norm_obs)), mode='constant')
        else:
            state = norm_obs[:expected_dim]
        action, _ = model.predict(state, deterministic=True)
        print(f"  Sample {i} -> action bias: {action[0]:+.4f}")

except Exception as e:
    print("Error:", e)
