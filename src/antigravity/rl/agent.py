import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces

class KellyConvexEnv(gym.Env):
    def __init__(self, data_stream, max_leverage=3.0, max_episode_steps=1000, target_dim=9, sharpe_lambda=0.0, drawdown_lambda=0.0, spread_pct=0.0020):
        """
        spread_pct : float
            One-way transaction cost as a fraction of notional (default 0.20%).
            Charged ONCE on position entry and ONCE on position exit — matching
            Alpaca's taker-fee model.  NOT a per-bar holding tax.

            Correct model:
              - Cash → Long  : pay spread_pct (entry)
              - Long  → Cash : pay spread_pct (exit)
              - Round trip   : 2 × spread_pct total

            A per-bar holding cost is NOT used because at 15m cadence it would
            imply 96 × 0.20% = 19.2%/day in fees, which is fictitious and causes
            immediate portfolio ruin rather than teaching the agent to be selective.
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
        self.spread_pct = spread_pct  # one-way cost fraction (charged on position change)
        self.returns_window = []
        self.peak_portfolio_value = 1.0
        self._prev_bias = 0  # tracks previous step's bias to detect entry/exit transitions
        
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

        # ── Spread / Fee Penalty (per-trade, not per-bar) ─────────────────────
        # Pay spread_pct once on ENTRY (cash → position) and once on EXIT
        # (position → cash or direction change).  This mirrors Alpaca's taker
        # fee model: you pay when you transact, not every bar you hold.
        #
        # Transitions that incur a cost:
        #   prev_bias == 0  and  bias != 0  →  entry into a position
        #   prev_bias != 0  and  bias == 0  →  exit to cash
        #   prev_bias != 0  and  bias != prev_bias  →  reversal (entry + exit)
        if self.spread_pct > 0.0 and abs(fraction_to_risk) > 1e-6:
            prev = self._prev_bias
            entering = (prev == 0 and bias != 0)
            exiting  = (prev != 0 and bias == 0)
            reversing = (prev != 0 and bias != 0 and bias != prev)
            if entering or exiting:
                portfolio_return -= abs(fraction_to_risk) * self.spread_pct
            elif reversing:
                # Round-trip cost: exit old side + enter new side
                portfolio_return -= abs(fraction_to_risk) * self.spread_pct * 2.0
        self._prev_bias = int(bias)
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
        self._prev_bias = 0  # start each episode from a flat/cash position
        
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