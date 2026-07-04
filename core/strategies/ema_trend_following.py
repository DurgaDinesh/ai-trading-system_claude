"""Turtle-style breakout confirmed by EMA stack alignment."""
import pandas as pd

from core.analysis.technical import get_ema_stack_signal, get_rsi_signal
from core.strategies.base import Strategy


class EMATrendFollowingStrategy(Strategy):
    name = "ema_trend_following"
    category = "trend"
    BREAKOUT_LOOKBACK = 20

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if len(df) < self.BREAKOUT_LOOKBACK + 1:
            return "NONE", 0.0, [], 0, "Insufficient bars for breakout lookback"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == 0:
            return "NONE", 0.0, [], 0, "EMA stack not aligned"

        close = df["close"].iloc[-1]
        lookback = df["close"].iloc[-(self.BREAKOUT_LOOKBACK + 1):-1]
        broke_high = close > lookback.max()
        broke_low = close < lookback.min()

        if ema_signal == 1 and broke_high:
            direction = "CE"
        elif ema_signal == -1 and broke_low:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No confirmed EMA-aligned breakout"

        indicators = ["EMA_STACK"]
        score = 65.0
        rsi_signal = get_rsi_signal(df)
        if (direction == "CE" and rsi_signal == 1) or (direction == "PE" and rsi_signal == -1):
            indicators.append("RSI")
            score += 15.0

        return direction, score, indicators, len(indicators), ""
