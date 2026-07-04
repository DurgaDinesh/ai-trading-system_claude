"""Trade the expansion that follows a period of contracted ATR (a volatility squeeze)."""
from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class ATRVolatilityContractionBreakoutStrategy(Strategy):
    name = "atr_volatility_contraction_breakout"
    category = "volatility"
    CONTRACTION_LOOKBACK = 14

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "atr" not in df.columns or len(df) < self.CONTRACTION_LOOKBACK + 2:
            return "NONE", 0.0, [], 0, "Insufficient ATR history"

        window = df["atr"].iloc[-(self.CONTRACTION_LOOKBACK + 2):-2]
        if window.empty:
            return "NONE", 0.0, [], 0, "Insufficient ATR history"

        atr_floor = window.min()
        prev_atr = df["atr"].iloc[-2]
        atr_now = df["atr"].iloc[-1]

        was_contracted = prev_atr <= atr_floor * 1.1
        expanding = atr_now > prev_atr * 1.15
        if not (was_contracted and expanding):
            return "NONE", 0.0, [], 0, "No ATR contraction-then-expansion"

        ema_signal = get_ema_stack_signal(df)
        if ema_signal == 1:
            direction = "CE"
        elif ema_signal == -1:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "ATR expanding but no directional bias"

        return direction, 63.0, ["ATR_EXPANSION", "EMA_STACK"], 2, ""
