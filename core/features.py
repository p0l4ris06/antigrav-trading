import polars as pl
import numpy as np

class SMCFeatureFactory:
    def __init__(self, swing_length=5):
        self.swing_length = swing_length

    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty(): return df
        df = self._calculate_base_metrics(df)
        df = self._identify_swings(df)
        df = self._calculate_structure(df)
        return df.drop_nulls()

    def _calculate_base_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
            (pl.col("high") - pl.col("low")).alias("true_range")
        ]).with_columns([
            (pl.col("true_range").rolling_mean(window_size=14) / pl.col("close")).alias("norm_atr")
        ])

    def _identify_swings(self, df: pl.DataFrame) -> pl.DataFrame:
        window_size = self.swing_length * 2 + 1
        df = df.with_columns([
            (pl.col("high") == pl.col("high").rolling_max(window_size=window_size, center=True)).alias("is_swing_high"),
            (pl.col("low") == pl.col("low").rolling_min(window_size=window_size, center=True)).alias("is_swing_low")
        ])
        return df.with_columns([
            pl.when(pl.col("is_swing_high")).then(pl.col("high")).forward_fill().alias("last_swing_high"),
            pl.when(pl.col("is_swing_low")).then(pl.col("low")).forward_fill().alias("last_swing_low")
        ])

    def _calculate_structure(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            ((pl.col("close") > pl.col("last_swing_high").shift(1)) & 
             (pl.col("last_swing_high") > pl.col("last_swing_high").shift(self.swing_length))).alias("bullish_bos"),
            ((pl.col("close") < pl.col("last_swing_low").shift(1)) & 
             (pl.col("last_swing_low") < pl.col("last_swing_low").shift(self.swing_length))).alias("bearish_bos"),
            ((pl.col("close") > pl.col("last_swing_high").shift(1)) & 
             (pl.col("last_swing_low") < pl.col("last_swing_low").shift(self.swing_length))).alias("bullish_choch"),
            ((pl.col("close") < pl.col("last_swing_low").shift(1)) & 
             (pl.col("last_swing_high") > pl.col("last_swing_high").shift(self.swing_length))).alias("bearish_choch")
        ])
        bool_cols = ["bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch"]
        return df.with_columns([pl.col(c).cast(pl.Float32) for c in bool_cols])
