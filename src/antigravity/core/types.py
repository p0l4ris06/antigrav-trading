from enum import Enum
from typing import TypedDict, Optional

class MarketRegime(str, Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    TREND_FOLLOWING = "TREND_FOLLOWING"
    VOLATILITY_BREAKOUT = "VOLATILITY_BREAKOUT"

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class Tick(TypedDict):
    timestamp: int
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float

class FeatureVector(TypedDict):
    obi: float
    spread: float
    atr: float
    vol_20: float
    vol_50: float
    vwap_60s: float
    vwap_300s: float
    volume_ratio: float
    log_returns: float
    spread_atr_norm: float
    microprice_deviation: float
