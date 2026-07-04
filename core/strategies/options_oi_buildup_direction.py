"""Follow the option chain's OI buildup signal (long buildup vs short covering)."""
from core.strategies.base import Strategy


class OptionsOIBuildupDirectionStrategy(Strategy):
    name = "options_oi_buildup_direction"
    category = "flow"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        oi_signal = options_context.get("oi_signal")
        if not oi_signal:
            return "NONE", 0.0, [], 0, "No OI buildup signal"

        direction = "CE" if oi_signal == 1 else "PE"
        return direction, 60.0, ["OI_BUILDUP"], 1, ""
