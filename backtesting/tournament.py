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


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_composite_scores(results: list[dict], weights: dict, pf_cap: float) -> list[dict]:
    """Min-max normalize each metric across ranked strategies, blend by weights.
    Drawdown is inverted (lower is better). Mutates and returns `results`."""
    ranked = [r for r in results if r["status"] == "ranked"]
    if not ranked:
        return results

    pfs = _minmax([min(r["profit_factor"], pf_cap) for r in ranked])
    sharpes = _minmax([r["sharpe_approx"] for r in ranked])
    win_rates = _minmax([r["win_rate"] for r in ranked])
    drawdowns = _minmax([r["max_drawdown"] for r in ranked])

    for r, pf, sh, wr, dd in zip(ranked, pfs, sharpes, win_rates, drawdowns):
        r["composite_score"] = (
            weights["profit_factor"] * pf
            + weights["sharpe"] * sh
            + weights["win_rate"] * wr
            + weights["max_drawdown"] * (1.0 - dd)
        )
    return results


def run_tournament(
    strategies: list | None = None,
    df: pd.DataFrame | None = None,
    session_factory=None,
    now: datetime | None = None,
) -> list[dict]:
    """Backtest every strategy, rank, persist StrategyRanking rows, promote top N.

    All parameters are injectable for tests; production callers pass nothing.
    """
    from backtesting.backtest_engine import _apply_atr_levels
    from backtesting.simulation import simulate_signals
    from core.analysis.technical import compute_all
    from database.models import StrategyRanking

    t_cfg = _cfg["strategy_tournament"]
    now = now or datetime.utcnow()
    lookback_days = t_cfg["backtest_lookback_days"]
    period_end = now
    period_start = now - timedelta(days=lookback_days)

    if strategies is None:
        from core.strategies.registry import get_all_strategies
        strategies = get_all_strategies()

    if df is None:
        from core.market_data.historical import fetch_historical_yfinance
        raw = fetch_historical_yfinance(
            "NIFTY",
            start=period_start.date(),
            end=period_end.date(),
            interval=t_cfg.get("backtest_interval", "1h"),
        )
        if raw.empty:
            logger.error("tournament_no_data")
            return []
        df = compute_all(raw)

    if session_factory is None:
        from database.trade_journal import SessionLocal, init_db
        init_db()  # ensures the new strategy_rankings table exists
        session_factory = SessionLocal

    initial_capital = _cfg["backtesting"]["initial_capital"]
    min_trades = t_cfg["min_backtest_trades_for_ranking"]
    period_days = max((df.index[-1] - df.index[0]).days, 1)

    regimes = precompute_regimes(df)
    results: list[dict] = []

    for strategy in strategies:
        base = {
            "strategy_name": strategy.name, "category": strategy.category,
            "win_rate": 0.0, "profit_factor": 0.0, "sharpe_approx": 0.0,
            "max_drawdown": 0.0, "opportunity_capture_rate": 0.0,
            "composite_score": 0.0, "rank": None, "promoted": False,
        }
        try:
            signals = generate_strategy_signals(strategy, df, regimes)
            sl_s, tp1_s, tp2_s = _apply_atr_levels(df, signals)
            sim = simulate_signals(df, signals, sl_s, tp1_s, tp2_s,
                                   initial_capital, period_days)
            if sim["total_trades"] < min_trades:
                results.append({**base, "status": "insufficient_data"})
                logger.info("tournament_insufficient_data",
                            strategy=strategy.name, trades=sim["total_trades"])
                continue
            results.append({
                **base, "status": "ranked",
                "win_rate": sim["win_rate"],
                "profit_factor": sim["profit_factor"],
                "sharpe_approx": sim["sharpe_ratio"],
                "max_drawdown": sim["max_drawdown"],
            })
        except Exception as e:
            logger.error("tournament_strategy_errored", strategy=strategy.name, error=str(e))
            results.append({**base, "status": "errored"})

    weights = t_cfg["score_weights"]
    pf_cap = t_cfg.get("profit_factor_cap", 10.0)
    compute_composite_scores(results, weights, pf_cap)

    ranked = sorted(
        [r for r in results if r["status"] == "ranked"],
        key=lambda r: r["composite_score"], reverse=True,
    )
    promote_top_n = t_cfg["promote_top_n"]
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
        r["promoted"] = i < promote_top_n

    db = session_factory()
    try:
        for r in results:
            db.add(StrategyRanking(period_start=period_start, period_end=period_end, **r))
        db.commit()
    finally:
        db.close()

    logger.info("tournament_complete",
                total=len(results),
                ranked=len(ranked),
                promoted=[r["strategy_name"] for r in ranked if r["promoted"]])
    return results


if __name__ == "__main__":
    rows = run_tournament()
    for r in sorted(rows, key=lambda x: (x["rank"] is None, x["rank"] or 0)):
        flag = "PROMOTED" if r["promoted"] else r["status"]
        print(f"{r['rank'] or '-':>3}  {r['strategy_name']:<40} "
              f"score={r['composite_score']:.3f}  {flag}")
