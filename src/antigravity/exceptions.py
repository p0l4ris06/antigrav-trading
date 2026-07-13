class AntigravityError(Exception):
    """Base exception for all Antigravity errors."""
    pass

class FeatureValidationError(AntigravityError):
    """Features failed validation (NaN, Inf, shape mismatch)."""
    pass

class ModelDimensionMismatch(AntigravityError):
    """Model observation space doesn't match feature output."""
    pass

class StateCorruptionError(AntigravityError):
    """Daemon state file is corrupted or locked."""
    pass

class EquityUnavailableError(AntigravityError):
    """Cannot determine account equity (no fallback)."""
    pass

class DataQualityError(AntigravityError):
    """Data validation failed (missing columns, invalid ranges)."""
    pass
