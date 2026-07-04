"""Trade in the direction of strong news/headline sentiment."""
from core.strategies.base import Strategy


class NewsSentimentMomentumStrategy(Strategy):
    name = "news_sentiment_momentum"
    category = "flow"
    MIN_SCORE = 0.4

    def evaluate(self, df, regime, options_context, global_context, news_sentiment):
        score = news_sentiment.get("score")
        if score is None:
            return "NONE", 0.0, [], 0, "News sentiment unavailable"

        if score >= self.MIN_SCORE:
            direction = "CE"
        elif score <= -self.MIN_SCORE:
            direction = "PE"
        else:
            return "NONE", 0.0, [], 0, "News sentiment not strong enough"

        return direction, 56.0, ["NEWS_SENTIMENT"], 1, ""
