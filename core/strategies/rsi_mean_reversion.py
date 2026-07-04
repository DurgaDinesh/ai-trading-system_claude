"""Classic RSI oversold/overbought reversal, confirmed by VWAP side."""
from core.analysis.technical import get_vwap_signal
from core.strategies.base import Strategy


class RSIMeanReversionStrategy(Strategy):
    name = "rsi_mean_reversion"
    category = "mean_reversion"
    OVERSOLD = 30
    OVERBOUGHT = 70

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "rsi" not in df.columns or df.empty:
            return "NONE", 0.0, [], 0, "RSI column missing"

        rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2] if len(df) > 1 else rsi

        if rsi < self.OVERSOLD and rsi > prev_rsi:
            direction = "CE"
        elif rsi > self.OVERBOUGHT and rsi < prev_rsi:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "RSI not in reversal zone"

        indicators = ["RSI"]
        score = 60.0
        vwap_signal = get_vwap_signal(df)
        if (direction == "CE" and vwap_signal >= 0) or (direction == "PE" and vwap_signal <= 0):
            indicators.append("VWAP")
            score += 10.0

        return direction, score, indicators, len(indicators), ""
