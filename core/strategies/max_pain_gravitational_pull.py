"""Price tends to drift toward the option chain's max-pain strike near expiry."""
from core.strategies.base import Strategy


class MaxPainGravitationalPullStrategy(Strategy):
    name = "max_pain_gravitational_pull"
    category = "flow"
    MIN_DEVIATION_PCT = 0.005

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        max_pain = options_context.get("max_pain")
        if max_pain is None or df.empty:
            return "NONE", 0.0, [], 0, "Max pain unavailable"

        close = df["close"].iloc[-1]
        if not close:
            return "NONE", 0.0, [], 0, "Invalid close price"

        deviation = (close - max_pain) / close
        if deviation > self.MIN_DEVIATION_PCT:
            direction = "PE"
        elif deviation < -self.MIN_DEVIATION_PCT:
            direction = "CE"
        else:
            return "NONE", 0.0, [], 0, "Price near max pain"

        return direction, 55.0, ["MAX_PAIN"], 1, ""
