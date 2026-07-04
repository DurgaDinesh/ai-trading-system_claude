"""
Weekly strategy tournament: backtests every registered strategy over a rolling
lookback window, ranks them by a config-weighted composite score, and marks
the top N as promoted. Orchestration entry point: run_tournament().
"""
from datetime import date, datetime, timedelta

import pandas as pd
import structlog
import yaml

from core.signals.regime_detector import detect_regime
from core.strategies.base import Strategy

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

WARMUP_BARS = 200  # same warmup the single-pipeline backtest uses


def neutral_contexts() -> tuple[dict, dict, dict]:
    """Neutral options/global/news contexts for historical replay.

    Historical values for PCR, OI, FII flow, VIX, news are not stored, so the
    replay feeds neutral values. Flow/news strategies therefore never fire in
    the tournament and end up status=insufficient_data — by design (spec:
    strategies with too few backtest trades are excluded from ranking).
    """
    options_context = {"pcr": 1.0, "max_pain": None, "oi_signal": 0, "chain_df": pd.DataFrame()}
    global_context = {
        "india_vix": 15.0, "vix_high_vol_regime": False, "vix_extreme": False,
        "fii_net_cash_cr": 0.0, "dii_net_cash_cr": 0.0, "global_score": 0,
        "gift_nifty_gap_pct": None,
    }
    news_sentiment = {"score": 0.0, "summary": "historical replay", "risk_events": [],
                      "method": "neutral_replay"}
    return options_context, global_context, news_sentiment


def precompute_regimes(df: pd.DataFrame, warmup: int = WARMUP_BARS) -> list:
    """One regime per bar, shared by all 20 strategies (computed once)."""
    options_context, global_context, news_sentiment = neutral_contexts()
    regimes: list = [None] * len(df)
    for i in range(warmup, len(df)):
        window = df.iloc[max(0, i - warmup):i + 1]
        try:
            regimes[i] = detect_regime(window, global_context, news_sentiment, options_context)
        except Exception as e:  # regime failure on one bar must not kill the run
            logger.warning("tournament_regime_failed", bar=str(df.index[i]), error=str(e))
            regimes[i] = None
    return regimes


def generate_strategy_signals(
    strategy: Strategy,
    df: pd.DataFrame,
    regimes: list,
    warmup: int = WARMUP_BARS,
) -> pd.Series:
    """Replay one strategy over the enriched df. Exceptions propagate to the caller."""
    options_context, global_context, news_sentiment = neutral_contexts()
    signals = pd.Series(0, index=df.index)
    for i in range(warmup, len(df)):
        regime = regimes[i]
        if regime is None:
            continue
        window = df.iloc[max(0, i - warmup):i + 1]
        direction, _score, _inds, _conf, _reason = strategy.evaluate(
            window, regime, options_context, global_context, news_sentiment
        )
        if direction == "CE":
            signals.iloc[i] = 1
        elif direction == "PE":
            signals.iloc[i] = -1
    return signals
