"""Stochastic %K/%D crossover from oversold/overbought extremes."""
from core.analysis.technical import get_stochastic_signal
from core.strategies.base import Strategy


class StochasticReversalStrategy(Strategy):
    name = "stochastic_reversal"
    category = "mean_reversion"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        stoch_signal = get_stochastic_signal(df)
        if stoch_signal == 0:
            return "NONE", 0.0, [], 0, "No stochastic reversal"

        direction = "CE" if stoch_signal == 1 else "PE"
        return direction, 57.0, ["STOCHASTIC"], 1, ""
