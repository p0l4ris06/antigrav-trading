import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO


class KellyConvexEnv(gym.Env):
    def __init__(self, data_stream, max_leverage=3.0):
        super().__init__()
        self.data = data_stream
        self.max_leverage = max_leverage
        self.current_step = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.portfolio_value = 1.0

    def step(self, action):
        obs, current_price, atr = self._get_next_data()

        bias = np.sign(action[0])
        fraction_to_risk = action[1] * self.max_leverage

        simulated_asset_return = bias * (np.random.normal(0, atr) / current_price)
        portfolio_return = fraction_to_risk * simulated_asset_return

        self.portfolio_value *= (1 + portfolio_return)

        clipped_return = max(portfolio_return, -0.999)
        reward = np.log(1 + clipped_return)

        bull_bos = obs[-4]
        if bull_bos > 0.5 and bias < 0:
            reward -= 0.01

        done = self.current_step >= len(self.data) - 1 or self.portfolio_value < 0.1
        return obs, reward, done, False, {"portfolio_value": self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 1.0
        obs, _, _ = self._get_next_data()
        return obs, {}

    def _get_next_data(self):
        self.current_step += 1
        return np.random.randn(15).astype(np.float32), 4400.00, 15.0


def init_agent():
    env = KellyConvexEnv(data_stream=np.zeros((1000, 15)))
    model = PPO("MlpPolicy", env, clip_range=0.2, ent_coef=0.01, learning_rate=3e-4)
    return model, env