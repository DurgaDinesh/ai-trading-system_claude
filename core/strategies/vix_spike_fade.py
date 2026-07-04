"""Fade the prevailing short-term trend when VIX signals panic/euphoria extremes."""
from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class VIXSpikeFadeStrategy(Strategy):
    name = "vix_spike_fade"
    category = "volatility"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if not global_context.get("vix_extreme"):
            return "NONE", 0.0, [], 0, "VIX not at extreme"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == -1:
            direction = "CE"
        elif ema_signal == 1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No fade direction confirmation from EMA stack"

        return direction, 55.0, ["VIX_EXTREME", "EMA_STACK"], 2, ""
