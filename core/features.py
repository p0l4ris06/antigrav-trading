import polars as pl


class SMCFeatureFactory:
    def __init__(self, swing_length=5):
        self.swing_length = swing_length

    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        df = self._calculate_base_metrics(df)
        df = self._identify_swings(df)
        df = self._calculate_structure(df)
        return df.drop_nulls()

    def _calculate_base_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        prev_close = pl.col("close").shift(1)
        prev_high = pl.col("high").shift(1)
        prev_low = pl.col("low").shift(1)

        eps = 1e-12
        safe_prev_close = prev_close.clip(lower_bound=eps)

        log_return = (pl.col("close").clip(lower_bound=eps) / safe_prev_close).log()

        tr1 = pl.col("high") - pl.col("low")
        tr2 = (pl.col("high") - prev_close).abs()
        tr3 = (pl.col("low") - prev_close).abs()
        true_range = pl.max_horizontal([tr1, tr2, tr3])

        atr = true_range.rolling_mean(window_size=14, min_samples=14)
        atr_std = true_range.rolling_std(window_size=14, min_samples=14)

        norm_atr = (atr / pl.col("close").clip(lower_bound=eps)).fill_nan(0.0).fill_null(0.0)
        norm_atr = (norm_atr / (1.0 + atr_std / (atr + eps))).fill_nan(0.0).fill_null(0.0)

        # --- Rich Alpha Seed Continuous Oscillators ---
        delta = pl.col("close").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

        avg_gain = gain.rolling_mean(window_size=14, min_samples=1)
        avg_loss = loss.rolling_mean(window_size=14, min_samples=1)
        rs = avg_gain / (avg_loss + eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        norm_rsi = (rsi / 100.0).fill_nan(0.5).fill_null(0.5)

        bb_mean = pl.col("close").rolling_mean(window_size=20, min_samples=1)
        bb_std = pl.col("close").rolling_std(window_size=20, min_samples=1).fill_null(0.0)
        norm_bb_width = (4.0 * bb_std / (bb_mean + eps)).fill_nan(0.0).fill_null(0.0)

        return df.with_columns([
            log_return.alias("log_return"),
            true_range.alias("true_range"),
            norm_atr.alias("norm_atr"),
            norm_rsi.alias("norm_rsi"),
            norm_bb_width.alias("norm_bb_width"),
        ])

    def _identify_swings(self, df: pl.DataFrame) -> pl.DataFrame:
        left = self.swing_length
        right = self.swing_length
        window_size = left + right + 1

        high = pl.col("high")
        low = pl.col("low")

        rolling_high = high.rolling_max(window_size=window_size, min_samples=window_size)
        rolling_low = low.rolling_min(window_size=window_size, min_samples=window_size)

        is_swing_high = (high == rolling_high)
        is_swing_low = (low == rolling_low)

        last_swing_high = (
            pl.when(pl.col("is_swing_high"))
            .then(pl.col("high"))
            .otherwise(None)
            .forward_fill()
        )
        last_swing_low = (
            pl.when(pl.col("is_swing_low"))
            .then(pl.col("low"))
            .otherwise(None)
            .forward_fill()
        )

        return df.with_columns([
            is_swing_high.alias("is_swing_high"),
            is_swing_low.alias("is_swing_low"),
        ]).with_columns([
            last_swing_high.alias("last_swing_high"),
            last_swing_low.alias("last_swing_low"),
        ])

    def _calculate_structure(self, df: pl.DataFrame) -> pl.DataFrame:
        last_high_prev = pl.col("last_swing_high").shift(1)
        last_low_prev = pl.col("last_swing_low").shift(1)

        swing_high_prev = pl.col("last_swing_high").shift(self.swing_length)
        swing_low_prev = pl.col("last_swing_low").shift(self.swing_length)

        hh = pl.col("last_swing_high") > swing_high_prev
        ll = pl.col("last_swing_low") < swing_low_prev

        close_above = pl.col("close") > last_high_prev
        close_below = pl.col("close") < last_low_prev

        bos_buffer = pl.col("true_range").rolling_mean(window_size=5, min_samples=1).fill_null(0.0)
        bos_up = pl.col("close") > (last_high_prev + 0.15 * bos_buffer)
        bos_down = pl.col("close") < (last_low_prev - 0.15 * bos_buffer)

        choch_up = pl.col("close") > (last_high_prev + 0.05 * bos_buffer)
        choch_down = pl.col("close") < (last_low_prev - 0.05 * bos_buffer)

        norm_swing_spread = ((pl.col("last_swing_high") - pl.col("last_swing_low")) / pl.col("close")).fill_nan(0.0).fill_null(0.0)

        df = df.with_columns([
            norm_swing_spread.alias("norm_swing_spread"),
            (bos_up & hh).alias("bullish_bos"),
            (bos_down & ll).alias("bearish_bos"),
            (choch_up & ll).alias("bullish_choch"),
            (choch_down & hh).alias("bearish_choch"),
        ])

        bool_cols = ["bullish_bos", "bearish_bos", "bullish_choch", "bearish_choch"]
        df = df.with_columns([pl.col(c).cast(pl.Float32) for c in bool_cols])
        return df.drop(["last_swing_high", "last_swing_low"])