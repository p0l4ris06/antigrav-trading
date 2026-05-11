import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class KellyConvexEnv(gym.Env):
    def __init__(self, data_stream, max_leverage=3.0, max_episode_steps=1000, target_dim=15):
        super().__init__()
        self.data = data_stream
        self.max_leverage = max_leverage
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.episode_steps = 0
        self.target_dim = target_dim
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.portfolio_value = 1.0 

    def step(self, action):
        obs, current_price, atr = self._get_next_data()
        self.episode_steps += 1
        
        bias = np.sign(action[0])
        fraction_to_risk = action[1] * self.max_leverage 
        
        # Use actual market return from log_return column (index 5) of the active candle
        # Scale by fractional Kelly risk coefficient (0.05) to ensure safety and prevent bankruptcy
        real_asset_return = float(obs[5]) if len(obs) > 5 else 0.0
        portfolio_return = fraction_to_risk * bias * real_asset_return * 0.05
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value = float(np.clip(self.portfolio_value, 1e-5, 1e9))
        
        # Reward: Logarithmic Wealth Utility
        clipped_return = max(portfolio_return, -0.999)
        reward = np.log(1 + clipped_return)
        
        # SMC Multiplier: Amplify rewards/penalties when aligned with structure
        bull_bos = obs[-4]
        bear_bos = obs[-3]
        alignment = (bias > 0 and bull_bos > 0.5) or (bias < 0 and bear_bos > 0.5)
        
        if alignment:
            reward *= 1.2 # Amplify reward/penalty when structurally aligned
        else:
            reward *= 0.8 # Attenuate signals that fight structure
        if bull_bos > 0.5 and bias < 0:
            reward -= 0.01 # Mild structural penalty
            
        done = (
            self.current_step >= len(self.data) - 1 or 
            self.portfolio_value < 0.1 or 
            self.episode_steps >= self.max_episode_steps
        )
        return obs, reward, done, False, {"portfolio_value": self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self.portfolio_value = 1.0 # FORCE RESET
        obs, _, _ = self._get_next_data()
        return obs, {}

    def _get_next_data(self):
        if self.current_step >= len(self.data):
            return np.zeros(15, dtype=np.float32), 4400.00, 15.0
        obs = self.data[self.current_step]
        current_price = float(obs[3]) if len(obs) > 3 and obs[3] > 0 else 4400.00
        atr = float(obs[6]) if len(obs) > 6 and obs[6] > 0 else 15.0
        self.current_step += 1
        return obs, current_price, atr

def init_agent():
    env = KellyConvexEnv(data_stream=np.zeros((1000, 15)))
    model = PPO("MlpPolicy", env, clip_range=0.2, ent_coef=0.01, learning_rate=3e-4)
    return model, env

