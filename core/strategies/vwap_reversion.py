"""Fade large intraday deviations from VWAP back toward it."""
from core.strategies.base import Strategy


class VWAPReversionStrategy(Strategy):
    name = "vwap_reversion"
    category = "mean_reversion"
    DEVIATION_PCT = 0.003

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "vwap" not in df.columns or df.empty:
            return "NONE", 0.0, [], 0, "VWAP column missing"

        close = df["close"].iloc[-1]
        vwap = df["vwap"].iloc[-1]
        if not vwap:
            return "NONE", 0.0, [], 0, "VWAP unavailable"

        deviation = (close - vwap) / vwap
        if deviation < -self.DEVIATION_PCT:
            direction = "CE"
        elif deviation > self.DEVIATION_PCT:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Price within VWAP band"

        score = 55.0 + min(abs(deviation) * 1000, 25.0)
        return direction, score, ["VWAP"], 1, ""
