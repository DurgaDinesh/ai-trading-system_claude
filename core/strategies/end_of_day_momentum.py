"""Trade last-hour trend continuation, confirmed by both MACD state and EMA stack.

Uses MACD *state* (above/below its signal line), not the crossover event —
a continuation strategy must fire while an established trend persists, and
get_macd_signal only returns nonzero on the single crossover bar.
"""
from datetime import time

import pandas as pd

from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class EndOfDayMomentumStrategy(Strategy):
    name = "end_of_day_momentum"
    category = "trend"
    WINDOW_START = time(13, 30)

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        current_time = df.index[-1].time()
        if current_time < self.WINDOW_START:
            return "NONE", 0.0, [], 0, "Outside end-of-day window"

        if not {"macd", "macd_signal"}.issubset(df.columns):
            return "NONE", 0.0, [], 0, "MACD columns missing"
        macd = df["macd"].iloc[-1]
        macd_sig = df["macd_signal"].iloc[-1]
        if pd.isna(macd) or pd.isna(macd_sig):
            return "NONE", 0.0, [], 0, "MACD not warmed up"
        macd_state = 1 if macd > macd_sig else (-1 if macd < macd_sig else 0)
        ema_signal = get_ema_stack_signal(df)

        if macd_state == 1 and ema_signal == 1:
            direction = "CE"
        elif macd_state == -1 and ema_signal == -1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No confirmed end-of-day continuation"

        return direction, 59.0, ["MACD", "EMA_STACK"], 2, ""
