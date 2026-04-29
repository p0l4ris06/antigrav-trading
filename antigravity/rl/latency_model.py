"""
Latency Predictor — Seq2Seq Future State Estimation.

Implements a Long Short-Term Memory (LSTM) network to map historical 
state sequences [s_{t-k}, ..., s_t] to a predicted future state 
s_{t+delta}, allowing the RL agent to act on anticipated market 
conditions rather than stale data.
"""

import torch
import torch.nn as nn
from typing import Tuple

class LatencyPredictor(nn.Module):
    """
    Seq2Seq model for microstructure feature forecasting.
    
    Architecture:
        - LSTM Encoder: Processes temporal dependencies in features
        - Linear Decoder: Projects hidden state to future feature vector
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Sequence of features (batch, seq_len, input_dim)
            hidden: Initial hidden/cell states
            
        Returns:
            predicted_state: Predicted features at t+delta (batch, input_dim)
            hidden: New hidden/cell states
        """
        # Encode the sequence
        out, (h, c) = self.lstm(x, hidden)
        
        # We only care about the last hidden state for prediction
        last_hidden = h[-1]
        
        # Decode to the future state vector
        prediction = self.decoder(last_hidden)
        
        return prediction, (h, c)

    def export_onnx(self, path: str, seq_len: int = 20):
        """Serialize for Rust inference."""
        dummy_input = torch.randn(1, seq_len, self.input_dim)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=["feature_history"],
            output_names=["predicted_state"],
            dynamic_axes={
                "feature_history": {0: "batch_size"},
                "predicted_state": {0: "batch_size"}
            },
            opset_version=17
        )
