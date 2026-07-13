import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces

class KellyConvexEnv(gym.Env):
    def __init__(self, data_stream, max_leverage=3.0, max_episode_steps=1000, target_dim=9, sharpe_lambda=0.0, drawdown_lambda=0.0, spread_pct=0.0020):
        """
        spread_pct : float
            One-way transaction cost as a fraction of notional (default 0.20%).
            Applied on every step where fraction_to_risk != 0 — i.e. the agent
            pays to hold any non-zero position each bar. This makes the training
            simulator more brutal than reality, forcing the agent to only trade
            when expected log-return genuinely exceeds the spread + fee cost.
            Alpaca taker fee ~0.15-0.25% per side → 0.0020 is a conservative mid.
        """
        super().__init__()
        self.data = data_stream
        self.max_leverage = max_leverage
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.episode_steps = 0
        self.target_dim = target_dim
        
        # Risk penalty parameters
        self.sharpe_lambda = sharpe_lambda
        self.drawdown_lambda = drawdown_lambda
        self.spread_pct = spread_pct  # one-way cost fraction
        self.returns_window = []
        self.peak_portfolio_value = 1.0
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.portfolio_value = 1.0 

    def step(self, action):
        obs, current_price, atr = self._get_next_data()
        self.episode_steps += 1
        
        bias = np.sign(action[0])
        fraction_to_risk = action[1] * self.max_leverage 
        
        # Use actual market return from log_return column of the active candle
        # Index 0 corresponds to log_return when raw prices are excluded (target_dim == 9)
        if self.target_dim == 9:
            real_asset_return = float(obs[0]) if len(obs) > 0 else 0.0
        else:
            real_asset_return = float(obs[5]) if len(obs) > 5 else 0.0
        portfolio_return = fraction_to_risk * bias * real_asset_return * 0.05

        # ── Spread / Fee Penalty ──────────────────────────────────────────────
        # Deduct one-way transaction cost proportional to position size on every
        # bar where the agent holds any non-zero exposure.  This forces the PPO
        # to only trade when E[log-return] > spread cost, eliminating the
        # frictionless hallucination that caused live dry-run bleed.
        if abs(fraction_to_risk) > 1e-6:
            spread_cost = abs(fraction_to_risk) * self.spread_pct
            portfolio_return -= spread_cost
        # ─────────────────────────────────────────────────────────────────────

        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value = float(np.clip(self.portfolio_value, 1e-5, 1e9))

        # Reward: Logarithmic Wealth Utility
        clipped_return = max(portfolio_return, -0.999)
        reward = float(np.log(1 + clipped_return))
        
        # SMC Multiplier: Asymmetric reward scaling
        bull_bos = obs[-4] if len(obs) >= 4 else 0.0
        bear_bos = obs[-3] if len(obs) >= 3 else 0.0
        alignment = (bias > 0 and bull_bos > 0.5) or (bias < 0 and bear_bos > 0.5)
        
        if alignment:
            # Good behavior: Amplify wins, cushion losses
            reward = reward * 1.2 if reward > 0 else reward * 0.8
        else:
            # Rogue behavior: Suppress wins, amplify losses
            reward = reward * 0.8 if reward > 0 else reward * 1.2
            
        if bull_bos > 0.5 and bias < 0:
            reward -= 0.01 # Mild structural penalty
            
        # Sharpe (return variance) penalty
        self.returns_window.append(portfolio_return)
        if len(self.returns_window) > 100:
            self.returns_window.pop(0)
            
        if self.sharpe_lambda > 0.0 and len(self.returns_window) > 10:
            returns_std = float(np.std(self.returns_window))
            reward -= self.sharpe_lambda * returns_std
            
        # Drawdown penalty
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            
        if self.drawdown_lambda > 0.0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            if drawdown > 0:
                reward -= self.drawdown_lambda * drawdown
            
        # FINAL CAP: Prevent numeric instability from reaching the optimizer
        reward = float(np.clip(reward, -10.0, 10.0))
            
        done = (
            self.current_step >= len(self.data) - 1 or 
            self.portfolio_value < 0.1 or 
            self.episode_steps >= self.max_episode_steps
        )
        return obs, reward, done, False, {"portfolio_value": self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self.returns_window = []
        self.peak_portfolio_value = 1.0
        
        # Randomize starting index during training to cover the entire dataset
        if self.max_episode_steps < len(self.data):
            self.current_step = np.random.randint(0, len(self.data) - self.max_episode_steps)
        else:
            self.current_step = 0
            
        self.portfolio_value = 1.0
        obs, _, _ = self._get_next_data()
        return obs, {}

    def _get_next_data(self):
        if self.current_step >= len(self.data):
            return np.zeros(self.target_dim, dtype=np.float32), 4400.00, 15.0
            
        obs = self.data[self.current_step]
        self.current_step += 1
        return obs, 4400.00, 15.0

def init_agent(target_dim=9, agent_type="ppo", policy_kwargs=None, env=None, data_stream=None, learning_rate=3e-4):
    if env is None:
        if data_stream is None:
            data_stream = np.zeros((1000, target_dim))
        env = KellyConvexEnv(data_stream=data_stream, target_dim=target_dim)
        
    if policy_kwargs is None:
        policy_kwargs = {}
        
    if agent_type.lower() == "ppo":
        from stable_baselines3 import PPO
        model = PPO(
            "MlpPolicy",
            env,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=learning_rate,
            n_steps=1024,
            batch_size=128,
            n_epochs=4,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu"
        )
    elif agent_type.lower() == "sac":
        from stable_baselines3 import SAC
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=10000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu"
        )
    elif agent_type.lower() == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=learning_rate,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu"
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
        
    return model, env

def load_agent_model(model_path, device="cpu"):
    from stable_baselines3 import PPO, SAC
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        RecurrentPPO = None
        
    # Attempt PPO
    try:
        return PPO.load(model_path, device=device)
    except Exception:
        pass
        
    # Attempt SAC
    try:
        return SAC.load(model_path, device=device)
    except Exception:
        pass
        
    # Attempt RecurrentPPO
    if RecurrentPPO is not None:
        try:
            return RecurrentPPO.load(model_path, device=device)
        except Exception:
            pass
            
    raise ValueError(f"Could not load model from {model_path} as PPO, SAC, or RecurrentPPO.")