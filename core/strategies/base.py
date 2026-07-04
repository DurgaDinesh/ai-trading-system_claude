"""Shared base class for all strategy archetypes in core/strategies/."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

from core.signals.signal_engine import TradeSignal, OBS_END, NO_NEW_TRADE_AFTER
from core.signals.regime_detector import RegimeResult

IST = pytz.timezone("Asia/Kolkata")


class Strategy(ABC):
    """
    A single directional (CE/PE) signal-generation archetype.

    Subclasses implement `evaluate()` with their pattern-detection logic only.
    `generate_signal()` (not overridden) handles session-time gating and
    TradeSignal construction, matching the shape of the existing
    core.signals.signal_engine.generate_signal function.
    """

    name: str = "unnamed_strategy"
    category: str = "uncategorized"

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        options_context: dict,
        global_context: dict,
        news_sentiment: dict,
    ) -> tuple[str, float, list[str], int, str]:
        """
        Detect this archetype's pattern on an already-enriched df (caller must
        have called core.analysis.technical.compute_all(df) first).

        Returns (direction, score, indicators_triggered, confluence_count, invalidation_reason).
        direction is "CE", "PE", or "NONE". When direction == "NONE",
        invalidation_reason must be a non-empty explanation.
        """
        raise NotImplementedError

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: RegimeResult,
        options_context: dict,
        global_context: dict,
        news_sentiment: dict,
        now: Optional[datetime] = None,
    ) -> TradeSignal:
        now = now or datetime.now(pytz.UTC)
        ist_time = now.astimezone(IST).time()

        if ist_time < OBS_END or ist_time >= NO_NEW_TRADE_AFTER:
            return self._invalid(regime, now, "Outside strategy trading window")

        direction, score, indicators, confluence, reason = self.evaluate(
            df, regime, options_context, global_context, news_sentiment
        )

        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and not df.empty else 0.0
        entry_price = float(df["close"].iloc[-1]) if not df.empty else 0.0

        if direction == "NONE" or not direction:
            return self._invalid(regime, now, reason or "No pattern match", entry_price, atr)

        return TradeSignal(
            direction=direction,
            composite_score=round(score, 1),
            confluence_count=confluence,
            indicators_triggered=indicators,
            entry_price=entry_price,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            atr=atr,
            rr_ratio=0.0,
            regime=regime.regime.value,
            strategy=self.name,
            timestamp=now,
            is_valid=True,
            invalidation_reason="",
        )

    def _invalid(
        self,
        regime: RegimeResult,
        now: datetime,
        reason: str,
        entry_price: float = 0.0,
        atr: float = 0.0,
    ) -> TradeSignal:
        return TradeSignal(
            direction="NONE",
            composite_score=0.0,
            confluence_count=0,
            indicators_triggered=[],
            entry_price=entry_price,
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            tp3=0.0,
            atr=atr,
            rr_ratio=0.0,
            regime=regime.regime.value,
            strategy=self.name,
            timestamp=now,
            is_valid=False,
            invalidation_reason=reason,
        )
