"""
Trading Environment — Custom Gymnasium Environment.

Maps the RL agent's continuous action space to capital allocation weights.
Reward function: R_t = (MFE_t - λ·MAE_t) / ATR_t

Design decisions:
    - Action space: Box([-1, 1], shape=(n_assets,)) → Softmax for Σ=1.0
    - Observation: concatenation of feature vector + regime probabilities
    - Episodes: rolling windows over historical tick data
    - Reward: ATR-normalized MFE/MAE (volatility-parity reward)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from antigravity.config import settings


class TradingEnv(gym.Env):
    """
    Gymnasium-compliant trading environment for RL training.

    State space:
        Concatenation of:
            - n_features normalized microstructure features
            - n_regimes GMM soft regime probabilities

    Action space:
        Continuous Box([-1, 1], shape=(n_assets,))
        Passed through softmax in step() to enforce Σ weights = 1.0

    Reward:
        R_t = (MFE_t - λ·MAE_t) / ATR_t
        Where MFE/MAE track peak unrealized profit/loss during each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_data: np.ndarray,
        price_data: np.ndarray,
        atr_data: np.ndarray,
        regime_data: np.ndarray | None = None,
        n_assets: int = 1,
        lambda_penalty: float | None = None,
        episode_length: int = 500,
        initial_capital: float = 100_000.0,
    ) -> None:
        """
        Args:
            feature_data: shape (T, n_features) — precomputed feature matrix
            price_data:   shape (T,) — last_price series for PnL calc
            atr_data:     shape (T,) — ATR series for reward normalization
            regime_data:  shape (T, n_regimes) — GMM probabilities (optional)
            n_assets:     number of assets to allocate across
            lambda_penalty: asymmetric MAE penalty (default from config)
            episode_length: steps per episode
            initial_capital: starting portfolio value
        """
        super().__init__()

        self._features = feature_data.astype(np.float32)
        self._prices = price_data.astype(np.float64)
        self._atr = atr_data.astype(np.float64)
        self._regimes = (
            regime_data.astype(np.float32)
            if regime_data is not None
            else np.zeros((len(feature_data), 3), dtype=np.float32)
        )

        self._n_assets = n_assets
        self._lambda = lambda_penalty or settings.rl.lambda_penalty
        self._episode_length = min(episode_length, len(feature_data) - 1)
        self._initial_capital = initial_capital

        # Observation: features + regime probabilities
        n_features = self._features.shape[1]
        n_regimes = self._regimes.shape[1]
        obs_dim = n_features + n_regimes

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: continuous allocation weights in [-1, 1], softmaxed in step()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        # Episode state
        self._t: int = 0
        self._start_idx: int = 0
        self._capital: float = initial_capital
        self._entry_price: float = 0.0
        self._position: float = 0.0
        self._mfe: float = 0.0
        self._mae: float = 0.0
        self._episode_returns: list[float] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a random starting point within the data."""
        super().reset(seed=seed)

        max_start = len(self._features) - self._episode_length - 1
        if max_start <= 0:
            self._start_idx = 0
        else:
            self._start_idx = self.np_random.integers(0, max_start)

        self._t = 0
        self._capital = self._initial_capital
        self._entry_price = self._prices[self._start_idx]
        self._position = 0.0
        self._mfe = 0.0
        self._mae = 0.0
        self._episode_returns = []

        obs = self._get_obs()
        info = {"capital": self._capital, "step": self._t}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step:
            1. Softmax-normalize action → portfolio weights
            2. Compute PnL from price change
            3. Track MFE (max profit) and MAE (max drawdown)
            4. Compute reward R_t = (MFE - λ·MAE) / ATR
        """
        self._t += 1
        idx = self._start_idx + self._t

        # Softmax normalization: ensures Σ weights = 1.0
        exp_a = np.exp(action - np.max(action))  # numerical stability
        weights = exp_a / (exp_a.sum() + 1e-10)

        # Price change
        prev_price = self._prices[idx - 1]
        curr_price = self._prices[idx]
        price_return = (curr_price - prev_price) / (prev_price + 1e-10)

        # Portfolio return (weighted sum across assets; single asset = weights[0])
        portfolio_return = float(np.sum(weights * price_return))
        self._capital *= (1 + portfolio_return)

        # MFE/MAE tracking
        unrealized_pnl = (curr_price - self._entry_price) / (self._entry_price + 1e-10)
        self._mfe = max(self._mfe, unrealized_pnl)
        self._mae = max(self._mae, -unrealized_pnl)

        # ATR for normalization
        atr = max(self._atr[idx], 1e-10)

        # Reward: R_t = (MFE - λ·MAE) / ATR
        reward = float((self._mfe - self._lambda * self._mae) / atr)

        # Scale reward to reasonable range for PPO
        reward = np.clip(reward, -10.0, 10.0)

        self._episode_returns.append(portfolio_return)

        # Termination conditions
        terminated = False
        truncated = False

        # Terminate if capital drops below 50% (catastrophic drawdown)
        if self._capital < self._initial_capital * 0.5:
            terminated = True
            reward = -10.0  # severe penalty

        # Truncate at episode length
        if self._t >= self._episode_length:
            truncated = True

        # Truncate if we've exhausted the data
        if idx >= len(self._features) - 1:
            truncated = True

        obs = self._get_obs()
        info = {
            "capital": self._capital,
            "mfe": self._mfe,
            "mae": self._mae,
            "atr": atr,
            "weights": weights.tolist(),
            "portfolio_return": portfolio_return,
            "step": self._t,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector: features || regime probabilities."""
        idx = self._start_idx + self._t
        idx = min(idx, len(self._features) - 1)

        features = self._features[idx]
        regimes = self._regimes[idx]

        obs = np.concatenate([features, regimes]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def render(self) -> None:
        """Optional: print current state."""
        print(
            f"Step {self._t} | Capital: ${self._capital:,.2f} | "
            f"MFE: {self._mfe:.4f} | MAE: {self._mae:.4f}"
        )
