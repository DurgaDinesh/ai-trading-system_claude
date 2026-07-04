"""Trade in the direction of a strong global overnight cues composite score."""
from core.strategies.base import Strategy


class GlobalCuesGapTradingStrategy(Strategy):
    name = "global_cues_gap_trading"
    category = "flow"
    MIN_SCORE = 2

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        global_score = global_context.get("global_score")
        if global_score is None:
            return "NONE", 0.0, [], 0, "Global score unavailable"

        if global_score >= self.MIN_SCORE:
            direction = "CE"
        elif global_score <= -self.MIN_SCORE:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "Global score within neutral range"

        return direction, 58.0, ["GLOBAL_SCORE"], 1, ""
