"""Trade Supertrend direction flips, confirmed by EMA stack agreement.

Note the `ta`-library column contract (see technical._add_supertrend):
supertrend_dir == -1 means price ABOVE the supertrend line (bullish),
supertrend_dir == +1 means price BELOW it (bearish).
"""
import pandas as pd

from core.analysis.technical import get_ema_stack_signal
from core.strategies.base import Strategy


class SupertrendTrendFollowingStrategy(Strategy):
    name = "supertrend_trend_following"
    category = "trend"

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if "supertrend_dir" not in df.columns or len(df) < 2:
            return "NONE", 0.0, [], 0, "Insufficient Supertrend history"

        dir_now = df["supertrend_dir"].iloc[-1]
        dir_prev = df["supertrend_dir"].iloc[-2]
        if pd.isna(dir_now) or pd.isna(dir_prev) or dir_now == dir_prev:
            return "NONE", 0.0, [], 0, "No Supertrend flip"

        # dir -1 = bullish, +1 = bearish (ta-library convention)
        direction = "CE" if dir_now == -1 else "PE"
        indicators = ["SUPERTREND"]
        score = 65.0

        ema_signal = get_ema_stack_signal(df)
        if (direction == "CE" and ema_signal == 1) or (direction == "PE" and ema_signal == -1):
            indicators.append("EMA_STACK")
            score += 15.0

        return direction, score, indicators, len(indicators), ""
