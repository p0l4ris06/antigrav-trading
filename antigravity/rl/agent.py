"""
RL Agent Manager — PPO/SAC Wrapper with Shadow Fork Support.

Wraps Stable Baselines3 PPO with:
    - VecNormalize for observation/reward normalization
    - Weight cloning for shadow fork training
    - Thread-safe weight hot-swap
    - ONNX export for Rust inference migration
"""

from __future__ import annotations

import threading
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from antigravity.config import settings
from antigravity.rl.environment import TradingEnv
from antigravity.rl.latency_model import LatencyPredictor

logger = structlog.get_logger(__name__)


class AgentManager:
    """
    RL agent lifecycle manager.

    Responsibilities:
        - Initialize PPO with tuned hyperparameters
        - Train on historical data
        - Predict allocation weights (softmax-normalized)
        - Clone/load weights for shadow fork hot-swap
        - Export policy to ONNX
    """

    def __init__(
        self,
        env: TradingEnv,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._raw_env = env
        self._swap_lock = threading.Lock()

        # Vectorize and normalize
        self._vec_env = DummyVecEnv([lambda: env])
        self._vec_env = VecNormalize(
            self._vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        cfg = settings.rl

        _policy_kwargs = policy_kwargs or {
            "net_arch": dict(
                pi=cfg.net_arch_pi,
                vf=cfg.net_arch_vf,
            ),
            "activation_fn": torch.nn.ReLU,
        }

        self.model = PPO(
            policy="MlpPolicy",
            env=self._vec_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            policy_kwargs=_policy_kwargs,
            verbose=0,
            device="auto",
        )

        # Predictive Latency Modeling (Section 7)
        self._latency_cfg = settings.latency
        self._obs_history: deque[np.ndarray] = deque(maxlen=self._latency_cfg.history_len)
        self._latency_model = LatencyPredictor(
            input_dim=env.observation_space.shape[0],
            hidden_dim=self._latency_cfg.hidden_dim,
            num_layers=self._latency_cfg.num_layers,
        ).to(self.model.device)

        logger.info(
            "agent.initialized",
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            latency_predictor=self._latency_cfg.enabled,
            device=str(self.model.device),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 100_000, progress_bar: bool = False) -> None:
        """Train the PPO agent."""
        logger.info("agent.training_started", timesteps=total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
        )
        logger.info("agent.training_complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict portfolio allocation weights.

        Args:
            obs: observation vector (raw, pre-normalization)
            deterministic: if True, use mode of policy distribution

        Returns:
            Softmax-normalized weight vector summing to 1.0
        """
        # Store for latency model training
        self._obs_history.append(obs)

        # If latency prediction is enabled and we have enough history
        if self._latency_cfg.enabled and len(self._obs_history) == self._latency_cfg.history_len:
            obs = self._predict_future_state()

        with self._swap_lock:
            action, _ = self.model.predict(obs, deterministic=deterministic)

        # Softmax normalization
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / (exp_a.sum() + 1e-10)
        return weights

    def _predict_future_state(self) -> np.ndarray:
        """Use LSTM to project observation history into the future."""
        self._latency_model.eval()
        with torch.no_grad():
            # (1, history_len, obs_dim)
            x = torch.from_numpy(np.array(self._obs_history)).float()
            x = x.unsqueeze(0).to(self.model.device)
            
            pred, _ = self._latency_model(x)
            return pred.cpu().numpy().squeeze(0)

    def train_latency_model(self, obs_buffer: np.ndarray, epochs: int = 5) -> None:
        """
        Train the latency predictor on a buffer of historical observations.
        
        Args:
            obs_buffer: Array of shape (n_samples, obs_dim)
        """
        if len(obs_buffer) < self._latency_cfg.history_len + 1:
            return

        logger.info("agent.latency_training_started", samples=len(obs_buffer))
        self._latency_model.train()
        optimizer = torch.optim.Adam(
            self._latency_model.parameters(), 
            lr=self._latency_cfg.learning_rate
        )
        criterion = torch.nn.MSELoss()

        # Create overlapping sequences
        k = self._latency_cfg.history_len
        X, Y = [], []
        for i in range(len(obs_buffer) - k):
            X.append(obs_buffer[i : i + k])
            Y.append(obs_buffer[i + k])

        X_tensor = torch.from_numpy(np.array(X)).float().to(self.model.device)
        Y_tensor = torch.from_numpy(np.array(Y)).float().to(self.model.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred, _ = self._latency_model(X_tensor)
            loss = criterion(pred, Y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                logger.debug("agent.latency_loss", epoch=epoch+1, loss=float(loss))

        logger.info("agent.latency_training_complete", final_loss=float(loss))

    # ------------------------------------------------------------------
    # Shadow Fork — Weight Cloning & Hot-Swap
    # ------------------------------------------------------------------

    def clone_weights(self) -> dict[str, Any]:
        """
        Deep-copy current policy state_dict for shadow fork.
        Returns a serializable dict of tensors.
        """
        with self._swap_lock:
            state = deepcopy(self.model.policy.state_dict())
        logger.info("agent.weights_cloned")
        return state

    def load_weights(self, state_dict: dict[str, Any]) -> None:
        """
        Hot-swap policy weights (thread-safe).
        Called by the overseer after shadow fork validation.
        """
        with self._swap_lock:
            self.model.policy.load_state_dict(state_dict)
        logger.info("agent.weights_swapped")

    def save(self, path: str | Path) -> None:
        """Save the full model (policy + VecNormalize stats)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(p / "ppo_model"))
        self._vec_env.save(str(p / "vec_normalize.pkl"))
        logger.info("agent.saved", path=str(p))

    def load(self, path: str | Path) -> None:
        """Load a previously saved model."""
        p = Path(path)
        self.model = PPO.load(str(p / "ppo_model"), env=self._vec_env)
        self._vec_env = VecNormalize.load(
            str(p / "vec_normalize.pkl"), self._vec_env.venv
        )
        logger.info("agent.loaded", path=str(p))

    # ------------------------------------------------------------------
    # ONNX Export
    # ------------------------------------------------------------------

    def export_onnx(self, path: str | Path) -> Path:
        """
        Export the policy network to ONNX format.

        This produces a standalone inference graph that can be loaded
        by onnxruntime (Python) or tract/onnxruntime-rs (Rust).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        obs_dim = self._raw_env.observation_space.shape[0]
        dummy_input = torch.randn(1, obs_dim).to(self.model.device)

        self.model.policy.eval()
        torch.onnx.export(
            self.model.policy,
            dummy_input,
            str(out),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
            opset_version=17,
        )
        self.model.policy.train()

        logger.info("agent.onnx_exported", path=str(out))
        return out
