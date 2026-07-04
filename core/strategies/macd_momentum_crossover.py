"""MACD signal-line crossover, with a bonus for an expanding histogram."""
from core.analysis.technical import get_macd_signal
from core.strategies.base import Strategy


class MACDMomentumCrossoverStrategy(Strategy):
    name = "macd_momentum_crossover"
    category = "trend"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        macd_signal = get_macd_signal(df)
        if macd_signal == 0:
            return "NONE", 0.0, [], 0, "No MACD crossover"

        direction = "CE" if macd_signal == 1 else "PE"
        indicators = ["MACD"]
        score = 60.0

        if len(df) > 1 and "macd_hist" in df.columns:
            hist = df["macd_hist"].iloc[-1]
            prev_hist = df["macd_hist"].iloc[-2]
            if abs(hist) > abs(prev_hist):
                indicators.append("MACD_HIST_EXPANDING")
                score += 15.0

        return direction, score, indicators, len(indicators), ""
