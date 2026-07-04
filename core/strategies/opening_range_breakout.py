"""Break of the first 15 minutes' high/low, in either direction."""
from datetime import time

import pandas as pd

from core.strategies.base import Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    name = "opening_range_breakout"
    category = "breakout"
    RANGE_END = time(9, 30)

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        today = df.index[-1].date()
        range_bars = df.loc[(df.index.date == today) & (df.index.time <= self.RANGE_END)]
        if range_bars.empty:
            return "NONE", 0.0, [], 0, "Opening range not yet formed"

        orb_high = range_bars["high"].max()
        orb_low = range_bars["low"].min()
        close = df["close"].iloc[-1]

        if close > orb_high:
            direction = "CE"
        elif close < orb_low:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Price inside opening range"

        return direction, 62.0, ["OPENING_RANGE_BREAKOUT"], 1, ""
