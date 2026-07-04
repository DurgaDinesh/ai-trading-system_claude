"""Enter RSI pullbacks only when the detected regime confidently agrees with direction."""
from core.strategies.base import Strategy


class RegimeAlignedPullbackStrategy(Strategy):
    name = "regime_aligned_pullback"
    category = "trend"
    MIN_CONFIDENCE = 0.5

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if regime.direction == 0 or regime.confidence < self.MIN_CONFIDENCE:
            return "NONE", 0.0, [], 0, "No confident directional regime"

        rsi = df["rsi"].iloc[-1] if "rsi" in df.columns and not df.empty else 50.0
        prev_rsi = df["rsi"].iloc[-2] if "rsi" in df.columns and len(df) > 1 else rsi

        if regime.direction == 1 and rsi < 35 and rsi > prev_rsi:
            direction = "CE"
        elif regime.direction == -1 and rsi > 65 and rsi < prev_rsi:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No pullback entry confirmation aligned with regime"

        score = 50.0 + regime.confidence * 30.0
        return direction, score, ["REGIME_ALIGNMENT", "RSI"], 2, ""
