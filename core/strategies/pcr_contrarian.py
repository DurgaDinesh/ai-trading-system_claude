"""Fade extreme Put-Call Ratio readings."""
from core.strategies.base import Strategy


class PCRContrarianStrategy(Strategy):
    name = "pcr_contrarian"
    category = "flow"
    HIGH_PCR = 1.5
    LOW_PCR = 0.6

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        pcr = options_context.get("pcr")
        if pcr is None:
            return "NONE", 0.0, [], 0, "PCR unavailable"

        if pcr >= self.HIGH_PCR:
            direction = "CE"
        elif pcr <= self.LOW_PCR:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "PCR within neutral band"

        return direction, 58.0, ["PCR"], 1, ""
