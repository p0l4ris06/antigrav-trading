
import os
import polars as pl
import numpy as np
from stable_baselines3 import PPO
from core.features import SMCFeatureFactory
from core.agent import KellyConvexEnv

class AntigravBacktester:
    """
    High-fidelity backtester for Antigrav PPO agents.
    Mirrors KellyConvexEnv logic and SMCFeatureFactory exactly.
    """
    def __init__(self, model_path="models/ppo_antigrav_latest.zip", symbol="BTC/USDT"):
        self.model_path = model_path
        self.symbol = symbol
        self.factory = SMCFeatureFactory()
        self.model = None
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, device='cpu')
            print(f"Loaded model from {self.model_path} (Forced CPU)")
        else:
            print(f"Model not found at {self.model_path}")
            
    def run(self, data_path):
        print(f"Running backtest on {data_path}...")
        df = pl.read_parquet(data_path)
        features_df = self.factory.compute_features(df)
        
        # Convert to numeric features array excluding non-stationary columns
        exclude_cols = {"open", "high", "low", "close", "volume", "true_range"}
        numeric_cols = [c for c, t in features_df.schema.items() if t in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and c not in exclude_cols]
        features_np = features_df.select(numeric_cols).to_numpy().astype(np.float32)
        
        # Ensure dynamic dim shape based on model
        target_dim = self.model.observation_space.shape[0] if self.model else 15
        if features_np.shape[1] < target_dim:
            padding = np.zeros((features_np.shape[0], target_dim - features_np.shape[1]), dtype=np.float32)
            features_np = np.hstack([features_np, padding])
        else:
            features_np = features_np[:, :target_dim]
            
        features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Setup Env
        env = KellyConvexEnv(data_stream=features_np, max_leverage=3.0, max_episode_steps=len(features_np), target_dim=target_dim)
        
        # Load VecNormalize stats if present
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        import os
        
        vec_norm_path = "models/vec_normalize.pkl"
        vec_norm = None
        if os.path.exists(vec_norm_path):
            try:
                dummy_env = DummyVecEnv([lambda: env])
                vec_norm = VecNormalize.load(vec_norm_path, dummy_env)
                vec_norm.training = False
                vec_norm.norm_reward = False
            except Exception as e:
                print(f"Warning: Failed to load VecNormalize: {e}")

        if vec_norm:
            obs = vec_norm.reset()
        else:
            obs, _ = env.reset()
            
        done = False
        equity_curve = [env.portfolio_value]
        actions = []
        
        def adapt_action(action, use_vec_env: bool) -> np.ndarray:
            action = np.asarray(action, dtype=np.float32)
            if action.ndim > 1:
                if action.shape[0] == 1:
                    action = action[0]
                else:
                    action = action[0]
            if action.ndim == 0:
                bias = float(action)
                action = np.array([bias, abs(bias)], dtype=np.float32)
            elif action.shape == (1,):
                bias = float(action[0])
                action = np.array([bias, abs(bias)], dtype=np.float32)
            elif action.shape == (2,):
                pass
            else:
                if action.shape[0] > 2:
                    action = action[:2]
                elif action.shape[0] == 0:
                    action = np.array([0.0, 0.0], dtype=np.float32)
                elif action.shape[0] == 1:
                    action = np.array([action[0], abs(action[0])], dtype=np.float32)
            if use_vec_env:
                return np.expand_dims(action, axis=0)
            return action

        while not done:
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
                action = adapt_action(action, use_vec_env=(vec_norm is not None))
            else:
                action = env.action_space.sample()
                action = adapt_action(action, use_vec_env=(vec_norm is not None))
                
            if vec_norm:
                obs, reward, dones, infos = vec_norm.step(action)
                done = dones[0]
                info = infos[0]
            else:
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                
            equity_curve.append(info["portfolio_value"])
            actions.append(action)
            
        print("Backtest Complete.")
        self.report(equity_curve, actions)
        return equity_curve, actions

    def report(self, equity_curve, actions):
        final_return = (equity_curve[-1] - 1.0) * 100
        returns = np.diff(equity_curve)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(365 * 96) # Annualized 15m
        max_dd = 0
        peak = equity_curve[0]
        for v in equity_curve:
            if v > peak: peak = v
            dd = (peak - v) / peak
            if dd > max_dd: max_dd = dd
            
        print(f"--- Results for {self.symbol} ---")
        print(f"Total Return: {final_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print(f"Final Equity: {equity_curve[-1]:.4f}")

if __name__ == "__main__":
    backtester = AntigravBacktester()
    backtester.load_model()
    # Try one of the data files
    data_files = [f for f in os.listdir("data") if f.endswith(".parquet")]
    if data_files:
        backtester.run(os.path.join("data", data_files[0]))
