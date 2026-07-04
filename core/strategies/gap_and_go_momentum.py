"""Trade continuation in the direction of a significant overnight gap."""
import pandas as pd

from core.strategies.base import Strategy


class GapAndGoMomentumStrategy(Strategy):
    name = "gap_and_go_momentum"
    category = "breakout"
    MIN_GAP_PCT = 0.003

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return "NONE", 0.0, [], 0, "DataFrame index is not datetime"

        today = df.index[-1].date()
        todays_bars = df.loc[df.index.date == today]
        prior_bars = df.loc[df.index.date < today]
        if todays_bars.empty or prior_bars.empty:
            return "NONE", 0.0, [], 0, "Insufficient prior-day data for gap calc"

        prev_close = prior_bars["close"].iloc[-1]
        today_open = todays_bars["open"].iloc[0]
        if not prev_close:
            return "NONE", 0.0, [], 0, "Invalid previous close"

        gap_pct = (today_open - prev_close) / prev_close
        close = df["close"].iloc[-1]

        if gap_pct > self.MIN_GAP_PCT and close >= today_open:
            direction = "CE"
        elif gap_pct < -self.MIN_GAP_PCT and close <= today_open:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "No sustained gap-and-go"

        return direction, 60.0, ["GAP_AND_GO"], 1, ""
