"""
Phase 1 Self-Learning: Adaptive indicator weights.
After each trade, indicators that fired on a winning trade gain weight.
Indicators that fired on a losing trade lose weight.
Weights are normalized and persisted to the DB.
"""

import structlog
import yaml
from contextlib import contextmanager

from database.models import IndicatorWeight
from database.trade_journal import journal, get_db

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["adaptive_weights"]

INDICATOR_KEYS = ["ema_stack", "rsi", "macd", "vwap", "oi_analysis", "pcr", "global_market"]


def _get_current_weights() -> dict:
    """Load latest weights from DB. Falls back to config defaults."""
    try:
        with get_db() as db:
            row = db.query(IndicatorWeight).order_by(IndicatorWeight.id.desc()).first()
            if row:
                return {k: getattr(row, k) for k in INDICATOR_KEYS}
    except Exception as e:
        logger.warning("weights_load_failed", error=str(e))
    return dict(_cfg["initial"])


def _save_weights(weights: dict, total_trades: int, notes: str = ""):
    """Persist updated weights to DB."""
    with get_db() as db:
        row = IndicatorWeight(
            **{k: weights[k] for k in INDICATOR_KEYS},
            total_trades_used=total_trades,
            notes=notes,
        )
        db.add(row)
    logger.info("weights_saved", weights=weights, total_trades=total_trades)


def _normalize(weights: dict) -> dict:
    """
    Normalize weights to sum to 1.0 while respecting [min, max] on every
    weight. A single clamp-then-divide pass can push a weight back past its
    cap (e.g. a weight pinned at max, divided by a sum shrunk by clamping the
    others down to min, ends up above max again) — so clamp and renormalize
    repeatedly until the result is stable within bounds.
    """
    min_w = _cfg["min_weight"]
    max_w = _cfg["max_weight"]
    current = dict(weights)
    for _ in range(20):
        clamped = {k: max(min_w, min(max_w, v)) for k, v in current.items()}
        total = sum(clamped.values())
        if total <= 0:
            return clamped
        current = {k: v / total for k, v in clamped.items()}
        if all(min_w - 1e-9 <= v <= max_w + 1e-9 for v in current.values()):
            break
    return {k: round(v, 4) for k, v in current.items()}


def update_weights_on_trade_close(
    indicators_triggered: list[str],
    is_win: bool,
    trade_number: int,
):
    """
    Called after each trade closes.
    Winning trade: boost triggered indicator weights.
    Losing trade: reduce triggered indicator weights.
    """
    weights = _get_current_weights()
    step = _cfg["adjustment_step"]

    triggered_keys = set()
    for name in indicators_triggered:
        name_lower = name.lower()
        for key in INDICATOR_KEYS:
            if key.replace("_", "") in name_lower.replace("_", ""):
                triggered_keys.add(key)

    adjustment = step if is_win else -step
    for key in triggered_keys:
        weights[key] = weights.get(key, 0.10) + adjustment

    weights = _normalize(weights)

    # Periodic rebalance to prevent drift
    rebalance_n = _cfg.get("rebalance_every_n_trades", 10)
    if trade_number % rebalance_n == 0:
        weights = _normalize(weights)
        notes = f"Periodic rebalance at trade #{trade_number}"
    else:
        notes = f"Trade #{trade_number} — {'WIN' if is_win else 'LOSS'} — adjusted {list(triggered_keys)}"

    _save_weights(weights, trade_number, notes)
    return weights


def get_weight_history(n: int = 20) -> list[dict]:
    """Return last N weight snapshots for dashboard visualization."""
    try:
        with get_db() as db:
            rows = (
                db.query(IndicatorWeight)
                .order_by(IndicatorWeight.id.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "updated_at": r.updated_at.isoformat(),
                    **{k: getattr(r, k) for k in INDICATOR_KEYS},
                    "total_trades": r.total_trades_used,
                }
                for r in rows
            ]
    except Exception:
        return []
