"""Trade journal: persists every signal, decision, and outcome to the database."""

import uuid
from datetime import datetime, date
from typing import Optional
from contextlib import contextmanager

import yaml
import structlog
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Trade, DailyPnL, SystemEvent, IndicatorWeight, TradingMode

logger = structlog.get_logger(__name__)

_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["database"]
engine = create_engine(_cfg["url"], echo=_cfg.get("echo", False))
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    logger.info("database_initialized", url=_cfg["url"])


@contextmanager
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def new_trade_id() -> str:
    return f"TRD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


class TradeJournal:

    def create_trade(self, **kwargs) -> Trade:
        """Insert a new trade record and return it."""
        trade = Trade(trade_id=new_trade_id(), **kwargs)
        with get_db() as db:
            db.add(trade)
            db.flush()
            db.refresh(trade)
        logger.info("trade_created", trade_id=trade.trade_id, instrument=trade.instrument)
        return trade

    def update_trade(self, trade_id: str, **kwargs) -> Optional[Trade]:
        with get_db() as db:
            trade = db.query(Trade).filter(Trade.trade_id == trade_id).first()
            if not trade:
                logger.warning("trade_not_found", trade_id=trade_id)
                return None
            for k, v in kwargs.items():
                setattr(trade, k, v)
            trade.updated_at = datetime.utcnow()
        return trade

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        with get_db() as db:
            return db.query(Trade).filter(Trade.trade_id == trade_id).first()

    def get_open_trades(self, mode: TradingMode = None) -> list[Trade]:
        with get_db() as db:
            q = db.query(Trade).filter(Trade.status.in_(["OPEN", "PARTIAL"]))
            if mode:
                q = q.filter(Trade.mode == mode)
            return q.all()

    def get_trades_for_date(self, trading_date: date, mode: TradingMode = None) -> list[Trade]:
        with get_db() as db:
            q = db.query(Trade).filter(func.date(Trade.created_at) == trading_date)
            if mode:
                q = q.filter(Trade.mode == mode)
            return q.all()

    def get_trades_today(self, mode: TradingMode = None) -> list[Trade]:
        return self.get_trades_for_date(datetime.utcnow().date(), mode)

    def get_trades_for_range(self, start_date: date, end_date: date, mode: TradingMode = None) -> list[Trade]:
        with get_db() as db:
            q = db.query(Trade).filter(
                func.date(Trade.created_at) >= start_date,
                func.date(Trade.created_at) <= end_date,
            )
            if mode:
                q = q.filter(Trade.mode == mode)
            return q.all()

    def get_worst_single_loss(self, mode: TradingMode = None) -> float:
        """
        Worst single closed-trade P&L across the ENTIRE trade history (not
        just a recent window) — a live-escalation safety check that only
        looked at the most recent N trades would let an early catastrophic
        loss "age out" and silently stop counting against readiness.
        """
        with get_db() as db:
            q = db.query(func.min(Trade.net_pnl)).filter(Trade.status == "CLOSED")
            if mode:
                q = q.filter(Trade.mode == mode)
            worst = q.scalar()
            return worst if worst is not None else 0.0

    def get_recent_trades(self, n: int = 50, mode: TradingMode = None) -> list[Trade]:
        with get_db() as db:
            q = db.query(Trade).order_by(Trade.created_at.desc())
            if mode:
                q = q.filter(Trade.mode == mode)
            return q.limit(n).all()

    def compute_and_save_daily_pnl(self, trading_date: date, mode: TradingMode):
        trades = []
        with get_db() as db:
            trades = (
                db.query(Trade)
                .filter(
                    func.date(Trade.created_at) == trading_date,
                    Trade.status == "CLOSED",
                    Trade.mode == mode,
                )
                .all()
            )
        if not trades:
            return

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        gross_pnl = sum(t.gross_pnl for t in trades)
        net_pnl = sum(t.net_pnl for t in trades)
        gross_wins = sum(t.net_pnl for t in wins) if wins else 0
        gross_losses = abs(sum(t.net_pnl for t in losses))
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
        else:
            profit_factor = float("inf") if gross_wins > 0 else 0.0

        daily = DailyPnL(
            date=datetime.combine(trading_date, datetime.min.time()),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            brokerage=sum(t.brokerage for t in trades),
            win_rate=len(wins) / len(trades) if trades else 0,
            profit_factor=profit_factor,
            mode=mode,
        )
        with get_db() as db:
            existing = (
                db.query(DailyPnL)
                .filter(func.date(DailyPnL.date) == trading_date, DailyPnL.mode == mode)
                .first()
            )
            if existing:
                for attr in ["total_trades","winning_trades","losing_trades","gross_pnl",
                             "net_pnl","brokerage","win_rate","profit_factor"]:
                    setattr(existing, attr, getattr(daily, attr))
            else:
                db.add(daily)

    def log_event(self, event_type: str, description: str, severity: str = "INFO", metadata: dict = None):
        event = SystemEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            event_metadata=metadata or {},
        )
        with get_db() as db:
            db.add(event)
        logger.info("system_event", type=event_type, severity=severity, desc=description)

    def get_performance_stats(self, n_trades: int = None, mode: TradingMode = None) -> dict:
        with get_db() as db:
            q = db.query(Trade).filter(Trade.status == "CLOSED")
            if mode:
                q = q.filter(Trade.mode == mode)
            if n_trades:
                q = q.order_by(Trade.created_at.desc()).limit(n_trades)
            trades = q.all()

        if not trades:
            return {}

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        pnls = [t.net_pnl for t in trades]
        gross_wins = sum(t.net_pnl for t in wins) if wins else 0
        gross_losses = abs(sum(t.net_pnl for t in losses))
        if gross_losses > 0:
            profit_factor = gross_wins / gross_losses
        else:
            profit_factor = float("inf") if gross_wins > 0 else 0.0

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "profit_factor": profit_factor,
            "total_net_pnl": sum(pnls),
            "avg_pnl_per_trade": sum(pnls) / len(pnls),
            "max_win": max(pnls),
            "max_loss": min(pnls),
            "avg_win": gross_wins / len(wins) if wins else 0,
            "avg_loss": (gross_losses / len(losses)) * -1 if losses else 0,
            "sharpe_approx": _approx_sharpe(trades),
        }


def _approx_sharpe(trades: list) -> float:
    """
    Annualizes using this trade set's own actual frequency (trades per trading
    day, scaled to 252 trading days/year) rather than assuming one trade per
    day — this is an intraday system that can take several trades per day.
    """
    pnls = [t.net_pnl for t in trades]
    if len(pnls) < 2:
        return 0.0
    import statistics
    mean = statistics.mean(pnls)
    std = statistics.stdev(pnls)
    if not std:
        return 0.0
    timestamps = [t.entry_time or t.created_at for t in trades if t.entry_time or t.created_at]
    span_days = max((max(timestamps) - min(timestamps)).days, 1) if timestamps else 1
    trades_per_year = len(pnls) / span_days * 252
    return (mean / std) * (trades_per_year ** 0.5)


journal = TradeJournal()
