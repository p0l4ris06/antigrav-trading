import numpy as np
import logging

logger = logging.getLogger("antigravity.features.validation")

CANONICAL_FEATURES = [
    "log_return", "norm_atr", "norm_rsi", "norm_bb_width", "norm_swing_spread",
    "bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch",
    "trend_strength", "momentum_slope", "vol_regime", "candle_body_ratio",
    "obi", "volume_delta", "funding_rate"
]

def validate_features(arr: np.ndarray, expected_shape: tuple = (16,)) -> tuple[bool, str]:
    """Validate feature array before inference."""
    if arr.shape != expected_shape:
        return False, f"Shape mismatch: expected {expected_shape}, got {arr.shape}"
    
    if not np.isfinite(arr).all():
        nan_mask = ~np.isfinite(arr)
        bad_indices = np.where(nan_mask)[0].tolist()
        bad_names = [CANONICAL_FEATURES[i] for i in bad_indices if i < len(CANONICAL_FEATURES)]
        return False, f"NaN/Inf detected at indices {bad_indices} ({bad_names})"
    
    return True, "OK"

def pad_features(arr: np.ndarray, target_dim: int = 16) -> np.ndarray:
    """Safely pad/truncate features to match model input dimension."""
    if arr.shape[0] < target_dim:
        padding = np.zeros((target_dim - arr.shape[0],), dtype=np.float32)
        return np.concatenate([arr, padding])
    elif arr.shape[0] > target_dim:
        return arr[:target_dim]
    return arr
