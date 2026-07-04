"""
Shared trade-metric calculations.

One implementation used by: the live-escalation stats (database/trade_journal.py),
the single-pipeline backtest (backtesting/), and the strategy tournament.
Annualization is passed in by the caller so each call site keeps its own
convention (calendar days for backtests, trading days for the live journal).
"""
import statistics

import numpy as np


def annualization_from_span(n_trades: int, span_days: float, days_per_year: float = 365.25) -> float:
    """sqrt(trades-per-year) scaling factor for a per-trade Sharpe/Sortino."""
    if n_trades <= 0 or span_days <= 0:
        return 0.0
    trades_per_year = n_trades / span_days * days_per_year
    return trades_per_year ** 0.5


def compute_trade_metrics(pnls: list[float], annualization: float = 0.0) -> dict:
    """All per-trade-P&L metrics the system reports, from a list of net P&Ls."""
    if not pnls:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "max_drawdown": 0.0,
            "sharpe": 0.0, "sortino": 0.0,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses))
    if gross_losses > 0:
        profit_factor = gross_wins / gross_losses
    else:
        profit_factor = float("inf") if gross_wins > 0 else 0.0

    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.max(running_max - cumulative))

    if len(pnls) > 1:
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = (mean_pnl / std_pnl) * annualization if std_pnl else 0.0
    else:
        sharpe = 0.0

    neg = [p for p in pnls if p < 0]
    if len(neg) > 1:
        sortino_std = statistics.stdev(neg)
        sortino = (statistics.mean(pnls) / sortino_std) * annualization if sortino_std else 0.0
    else:
        sortino = 0.0

    return {
        "total_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(pnls),
        "profit_factor": profit_factor,
        "total_net_pnl": sum(pnls),
        "avg_pnl_per_trade": sum(pnls) / len(pnls),
        "max_win": max(pnls),
        "max_loss": min(pnls),
        "avg_win": gross_wins / len(wins) if wins else 0.0,
        "avg_loss": (gross_losses / len(losses)) * -1 if losses else 0.0,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
    }
