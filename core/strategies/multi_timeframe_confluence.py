"""Require the 5m EMA stack signal to agree with a resampled 15m EMA stack signal."""
import pandas as pd

from core.analysis.technical import compute_all, get_ema_stack_signal
from core.strategies.base import Strategy


class MultiTimeframeConfluenceStrategy(Strategy):
    name = "multi_timeframe_confluence"
    category = "trend"
    HIGHER_TF = "15min"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 10:
            return "NONE", 0.0, [], 0, "Insufficient data for multi-timeframe resample"

        higher_df = df.resample(self.HIGHER_TF).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(higher_df) < 5:
            return "NONE", 0.0, [], 0, "Insufficient higher-timeframe bars"

        higher_df = compute_all(higher_df)
        higher_signal = get_ema_stack_signal(higher_df)
        lower_signal = get_ema_stack_signal(df)

        if higher_signal == 0 or higher_signal != lower_signal:
            return "NONE", 0.0, [], 0, "Timeframes not in agreement"

        direction = "CE" if higher_signal == 1 else "PE"
        return direction, 66.0, ["EMA_STACK_5M", "EMA_STACK_15M"], 2, ""
