"""
Performance tracker: monitors paper trading stats and evaluates
whether the system meets the live trading escalation criteria.
"""

import structlog
import yaml
from datetime import datetime
from typing import Optional

from database.trade_journal import journal
from database.models import TradingMode
from core.learning.adaptive_weights import update_weights_on_trade_close
from core.learning.ml_scorer import ml_scorer

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))


class PerformanceTracker:
    """
    Tracks key metrics after each trade closes.
    Evaluates live trading escalation criteria.
    Triggers adaptive weight updates and ML retraining.
    """

    def on_trade_closed(
        self,
        trade_id: str,
        net_pnl: float,
        indicators_triggered: list[str],
        entry_price: float,
        order_value: float,
    ):
        """Call this immediately after any trade closes."""
        is_win = net_pnl > 0
        stats = journal.get_performance_stats(mode=TradingMode.PAPER)
        trade_number = stats.get("total_trades", 1)

        # Phase 1: Adaptive weight update
        update_weights_on_trade_close(indicators_triggered, is_win, trade_number)

        # Per-strategy score learning (AI brain): nudge the closed trade's
        # strategy in the latest tournament ranking. No-op for hardcoded-
        # pipeline trades (their strategy field is a regime label, not an
        # archetype), and never allowed to break the trade-close flow.
        try:
            trade = journal.get_trade(trade_id)
            if trade and trade.strategy:
                from core.learning.mistake_analyzer import update_strategy_score_on_trade_close
                update_strategy_score_on_trade_close(trade.strategy, is_win)
        except Exception as e:
            logger.warning("strategy_score_update_failed", trade_id=trade_id, error=str(e))

        # Phase 2: Check if ML training should trigger
        if ml_scorer.should_train():
            logger.info("triggering_ml_retrain", n_trades=trade_number)
            try:
                ml_scorer.train()
            except Exception as e:
                logger.error("ml_retrain_failed", error=str(e))

        logger.info(
            "performance_updated",
            trade_id=trade_id,
            is_win=is_win,
            pnl=net_pnl,
            total_trades=trade_number,
            win_rate=stats.get("win_rate", 0),
        )

    def check_live_escalation_readiness(self) -> dict:
        """
        Evaluate all conditions for paper → live escalation.
        Returns a dict with status and individual criterion results.
        """
        criteria = _cfg["paper_to_live"]
        stats = journal.get_performance_stats(mode=TradingMode.PAPER)

        n_trades = stats.get("total_trades", 0)
        win_rate = stats.get("win_rate", 0)
        profit_factor = stats.get("profit_factor", 0)
        sharpe = stats.get("sharpe_approx", 0)

        # Max single loss check — scans the FULL paper trade history, not a
        # recent window, so an early catastrophic loss can never age out.
        worst_loss = journal.get_worst_single_loss(mode=TradingMode.PAPER)
        capital = _cfg["capital"]["total"]
        max_single_loss_pct = abs(worst_loss) / capital if worst_loss < 0 else 0

        results = {
            "min_trades": {
                "required": criteria["min_paper_trades"],
                "actual": n_trades,
                "passed": n_trades >= criteria["min_paper_trades"],
            },
            "win_rate": {
                "required": criteria["min_win_rate"],
                "actual": round(win_rate, 3),
                "passed": win_rate >= criteria["min_win_rate"],
            },
            "profit_factor": {
                "required": criteria["min_profit_factor"],
                "actual": round(profit_factor, 3),
                "passed": profit_factor >= criteria["min_profit_factor"],
            },
            "max_single_loss": {
                "required": criteria["max_single_loss_pct"],
                "actual": round(max_single_loss_pct, 3),
                "passed": max_single_loss_pct <= criteria["max_single_loss_pct"],
            },
            "sharpe_ratio": {
                "required": criteria["min_sharpe_ratio"],
                "actual": round(sharpe, 3),
                "passed": sharpe >= criteria["min_sharpe_ratio"],
            },
        }

        all_passed = all(r["passed"] for r in results.values())
        requires_human = criteria["require_human_approval"]

        summary = {
            "ready_for_live": all_passed,
            "requires_human_approval": requires_human,
            "human_approved": False,  # Must be set via dashboard toggle
            "criteria": results,
            "total_paper_trades": n_trades,
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        if all_passed:
            logger.info("LIVE_ESCALATION_CRITERIA_MET", stats=stats)
        else:
            failed = [k for k, v in results.items() if not v["passed"]]
            logger.info("live_escalation_not_ready", failed_criteria=failed)

        return summary

    def get_summary_report(self, n_trades: int = 50, mode: str = "paper") -> dict:
        stats = journal.get_performance_stats(n_trades=n_trades, mode=TradingMode(mode.upper()))
        from core.learning.adaptive_weights import _get_current_weights
        weights = _get_current_weights()
        return {
            "stats": stats,
            "current_weights": weights,
            "ml_active": ml_scorer.is_active,
            "live_readiness": self.check_live_escalation_readiness(),
        }


performance_tracker = PerformanceTracker()
