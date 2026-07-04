"""SQLAlchemy ORM models for the algo trading system."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, JSON, Enum, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class TradeDirection(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, enum.Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradingMode(str, enum.Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Instrument
    instrument = Column(String(50), nullable=False)    # e.g. NIFTY24JAN23500CE
    underlying = Column(String(20), nullable=False)    # NIFTY | BANKNIFTY
    exchange = Column(String(10), nullable=False)
    strike = Column(Float)
    option_type = Column(String(2))                    # CE | PE
    expiry = Column(DateTime)
    mode = Column(Enum(TradingMode), nullable=False)

    # Signal
    direction = Column(Enum(TradeDirection), nullable=False)
    composite_score = Column(Float)                    # 0–100
    confluence_count = Column(Integer)
    indicators_triggered = Column(JSON)                # list of indicator names
    regime = Column(String(30))                        # BULLISH_MOMENTUM etc.
    strategy = Column(String(50))

    # Entry
    signal_time = Column(DateTime)
    entry_time = Column(DateTime)
    entry_price = Column(Float)
    quantity = Column(Integer)
    remaining_quantity = Column(Integer)   # Decremented as partial TP legs are booked
    order_value = Column(Float)
    kite_order_id = Column(String(50))

    # Risk levels
    stop_loss = Column(Float)
    tp1 = Column(Float)
    tp2 = Column(Float)
    tp3 = Column(Float)
    atr_at_entry = Column(Float)

    # Exit
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))   # TP1 | TP2 | TP3 | SL | MANUAL | SQUAREOFF

    # Outcome
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING_APPROVAL)
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)            # after brokerage
    realized_rr = Column(Float)                      # actual RR achieved
    brokerage = Column(Float, default=0.0)

    # Context snapshot at signal time
    global_context = Column(JSON)                   # VIX, DXY, gift_nifty, FII etc.
    news_sentiment = Column(Float)                  # -1 to +1
    vix_at_signal = Column(Float)
    pcr_at_signal = Column(Float)

    # Approval workflow
    telegram_message_id = Column(String(50))
    approved_by = Column(String(20))               # HUMAN | AUTO | TIMEOUT_SKIP


class DailyPnL(Base):
    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("date", "mode", name="uq_daily_pnl_date_mode"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    brokerage = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    max_drawdown_intraday = Column(Float, default=0.0)
    mode = Column(Enum(TradingMode))


class IndicatorWeight(Base):
    __tablename__ = "indicator_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ema_stack = Column(Float, default=0.20)
    rsi = Column(Float, default=0.15)
    macd = Column(Float, default=0.15)
    vwap = Column(Float, default=0.15)
    oi_analysis = Column(Float, default=0.15)
    pcr = Column(Float, default=0.10)
    global_market = Column(Float, default=0.10)
    total_trades_used = Column(Integer, default=0)
    notes = Column(Text)


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    event_type = Column(String(50), nullable=False)   # HALT | RESUME | MODE_CHANGE | ANOMALY | ERROR
    severity = Column(String(10), default="INFO")      # INFO | WARNING | CRITICAL
    description = Column(Text)
    event_metadata = Column(JSON)


class MLModelMetric(Base):
    __tablename__ = "ml_model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trained_at = Column(DateTime, default=datetime.utcnow)
    model_type = Column(String(30))
    n_samples = Column(Integer)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    feature_importances = Column(JSON)
    model_path = Column(String(200))


class StrategyRanking(Base):
    """One row per strategy per tournament run (weekly)."""
    __tablename__ = "strategy_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    strategy_name = Column(String(60), nullable=False)
    category = Column(String(30))
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_approx = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    opportunity_capture_rate = Column(Float, default=0.0)  # wired by Plan 3
    composite_score = Column(Float, default=0.0)
    rank = Column(Integer)                                  # 1 = best; NULL if not ranked
    promoted = Column(Boolean, default=False)
    status = Column(String(20), default="ranked")           # ranked | insufficient_data | errored
