"""
Bar-by-bar signal simulation, extracted from backtest_engine.run_backtest so
the strategy tournament can reuse the exact same execution model.
"""
import numpy as np
import pandas as pd
import yaml

from core.learning.metrics import annualization_from_span, compute_trade_metrics

_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


def simulate_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    sl_series: pd.Series,
    tp1_series: pd.Series,
    tp2_series: pd.Series,
    initial_capital: float,
    period_days: float,
) -> dict:
    commission = _cfg["backtesting"]["commission_pct"]
    slippage = _cfg["backtesting"]["slippage_pct"]
    tp1_alloc = _cfg["signals"]["tp_allocation"]["tp1"]

    capital = initial_capital
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = None
    entry_sl = entry_tp1 = entry_tp2 = 0.0
    direction = 0
    tp1_booked = False
    remaining_fraction = 1.0

    # A signal on bar i can only be filled at bar i+1's open — the earliest
    # realistic execution point — never at the same bar's own close.
    pending_direction = None
    pending_sl = pending_tp1 = pending_tp2 = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        close = row["close"]
        idx = df.index[i]

        if pending_direction is not None and not in_trade:
            direction = pending_direction
            entry_price = row["open"] * (1 + slippage if direction == 1 else 1 - slippage)
            entry_sl = pending_sl
            entry_tp1 = pending_tp1
            entry_tp2 = pending_tp2
            entry_idx = idx
            in_trade = True
            tp1_booked = False
            remaining_fraction = 1.0
            pending_direction = None

        if in_trade:
            tp1_hit = close >= entry_tp1 if direction == 1 else close <= entry_tp1
            tp2_hit = close >= entry_tp2 if direction == 1 else close <= entry_tp2
            sl_hit = close <= entry_sl if direction == 1 else close >= entry_sl
            base_qty = capital * _cfg["capital"]["max_per_trade_pct"] / entry_price

            if sl_hit:
                fill = entry_sl * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "SL",
                })
                capital += net
                in_trade = False

            elif tp2_hit:
                fill = entry_tp2 * (1 - slippage if direction == 1 else 1 + slippage)
                pnl = (fill - entry_price) * direction
                gross = pnl * remaining_fraction * base_qty
                net = gross - gross * commission * 2
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": idx,
                    "direction": direction, "entry": entry_price, "exit": fill,
                    "pnl": net, "exit_reason": "TP2",
                })
                capital += net
                in_trade = False

            elif tp1_hit and not tp1_booked:
                # Book the configured TP1 fraction once, move SL to breakeven,
                # and leave the rest of the position open.
                tp1_booked = True
                entry_sl = entry_price
                fill = entry_tp1
                pnl = (fill - entry_price) * direction
                gross = pnl * tp1_alloc * base_qty
                net = gross - gross * commission
                capital += net
                remaining_fraction -= tp1_alloc

        elif signals.iloc[i] != 0 and not pd.isna(sl_series.iloc[i]) and pending_direction is None:
            pending_direction = int(signals.iloc[i])
            pending_sl = sl_series.iloc[i]
            pending_tp1 = tp1_series.iloc[i]
            pending_tp2 = tp2_series.iloc[i]

    if not trades:
        return {
            "final_capital": round(capital, 2), "total_return_pct": 0.0,
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "total_net_pnl": 0.0,
            "avg_pnl_per_trade": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "trades": [], "pnl_curve": [],
        }

    pnls = [t["pnl"] for t in trades]
    ann = annualization_from_span(len(pnls), max(period_days, 1))
    m = compute_trade_metrics(pnls, annualization=ann)
    cumulative = np.cumsum(pnls)

    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
        "total_trades": m["total_trades"],
        "winning_trades": m["winning_trades"],
        "losing_trades": m["losing_trades"],
        "win_rate": round(m["win_rate"], 3),
        "profit_factor": round(m["profit_factor"], 3) if m["profit_factor"] != float("inf") else float("inf"),
        "total_net_pnl": round(m["total_net_pnl"], 2),
        "avg_pnl_per_trade": round(m["avg_pnl_per_trade"], 2),
        "max_win": round(m["max_win"], 2),
        "max_loss": round(m["max_loss"], 2),
        "max_drawdown": round(m["max_drawdown"], 2),
        "max_drawdown_pct": round(m["max_drawdown"] / initial_capital * 100, 2),
        "sharpe_ratio": round(m["sharpe"], 3),
        "sortino_ratio": round(m["sortino"], 3),
        "trades": trades,
        "pnl_curve": cumulative.tolist(),
    }
