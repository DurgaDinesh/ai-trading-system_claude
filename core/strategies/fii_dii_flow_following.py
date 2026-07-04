"""Follow the direction of significant net FII cash flow."""
from core.strategies.base import Strategy


class FIIDIIFlowFollowingStrategy(Strategy):
    name = "fii_dii_flow_following"
    category = "flow"
    MIN_NET_CR = 500.0

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        fii = global_context.get("fii_net_cash_cr")
        if fii is None:
            return "NONE", 0.0, [], 0, "FII flow data unavailable"

        if fii >= self.MIN_NET_CR:
            direction = "CE"
        elif fii <= -self.MIN_NET_CR:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "FII flow below significance threshold"

        return direction, 57.0, ["FII_FLOW"], 1, ""
