"""Breakout of Bollinger Bands following a volatility squeeze."""
from core.analysis.technical import get_bollinger_signal
from core.strategies.base import Strategy


class BollingerSqueezeBreakoutStrategy(Strategy):
    name = "bollinger_squeeze_breakout"
    category = "breakout"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        bb_signal = get_bollinger_signal(df)
        if bb_signal == 0:
            return "NONE", 0.0, [], 0, "No squeeze breakout"

        direction = "CE" if bb_signal == 1 else "PE"
        return direction, 68.0, ["BOLLINGER_SQUEEZE"], 1, ""
