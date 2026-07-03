"""
Daily and weekly report generator.
Produces PDF trade journals and CSV data exports.
"""

import csv
import io
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import structlog
import yaml
import pytz

from database.trade_journal import journal, _approx_sharpe
from database.models import TradingMode

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
IST = pytz.timezone("Asia/Kolkata")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _format_inr(val: float) -> str:
    return f"₹{val:+,.2f}"


def _compute_stats_for_trades(trades: list) -> dict:
    """
    Stats scoped to exactly the given (already date-filtered) trade list —
    NOT a global "most recent N trades" query, which would silently pull in
    trades from other days once this date's own closed-trade count is small.
    """
    closed = [t for t in trades if t.status == "CLOSED"]
    if not closed:
        return {}
    wins = [t for t in closed if t.net_pnl > 0]
    losses = [t for t in closed if t.net_pnl <= 0]
    pnls = [t.net_pnl for t in closed]
    gross_wins = sum(t.net_pnl for t in wins) if wins else 0
    gross_losses = abs(sum(t.net_pnl for t in losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)
    return {
        "total_trades": len(closed),
        "win_rate": len(wins) / len(closed),
        "profit_factor": profit_factor,
        "total_net_pnl": sum(pnls),
        "avg_pnl_per_trade": sum(pnls) / len(pnls),
        "max_win": max(pnls),
        "max_loss": min(pnls),
        "sharpe_approx": _approx_sharpe(closed),
    }


def generate_daily_csv(trading_date: date, mode: str = "paper") -> Path:
    """Export all trades for a given date to CSV."""
    trades = journal.get_trades_for_date(trading_date, TradingMode(mode.upper()))
    filename = REPORTS_DIR / f"trades_{trading_date.strftime('%Y%m%d')}_{mode}.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trade ID", "Instrument", "Direction", "Entry Time", "Entry Price",
            "Exit Time", "Exit Price", "Qty", "SL", "TP1", "TP2", "TP3",
            "Gross P&L", "Net P&L", "Brokerage", "Exit Reason", "Score",
            "Confluence", "Indicators", "Regime", "VIX", "News Sentiment",
        ])
        for t in trades:
            writer.writerow([
                t.trade_id, t.instrument, t.option_type,
                t.entry_time.strftime("%H:%M:%S") if t.entry_time else "",
                t.entry_price,
                t.exit_time.strftime("%H:%M:%S") if t.exit_time else "",
                t.exit_price or "",
                t.quantity,
                t.stop_loss, t.tp1, t.tp2, t.tp3,
                t.gross_pnl or 0, t.net_pnl or 0, t.brokerage or 0,
                t.exit_reason or "",
                t.composite_score, t.confluence_count,
                ", ".join(t.indicators_triggered or []),
                t.regime, t.vix_at_signal, t.news_sentiment,
            ])

    logger.info("csv_report_generated", path=str(filename))
    return filename


def generate_daily_pdf(trading_date: date, mode: str = "paper") -> Optional[Path]:
    """Generate a PDF daily trade journal."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        logger.warning("reportlab_not_installed_skipping_pdf")
        return None

    trades = journal.get_trades_for_date(trading_date, TradingMode(mode.upper()))
    stats = _compute_stats_for_trades(trades)
    filename = REPORTS_DIR / f"report_{trading_date.strftime('%Y%m%d')}_{mode}.pdf"

    doc = SimpleDocTemplate(str(filename), pagesize=A4, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], alignment=TA_CENTER)
    elements.append(Paragraph(f"NiftySniper — Daily Trade Journal", title_style))
    elements.append(Paragraph(
        f"{trading_date.strftime('%d %B %Y')} | Mode: {mode.upper()} | Generated: {datetime.now(IST).strftime('%H:%M IST')}",
        ParagraphStyle("Sub", parent=styles["Normal"], alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 0.3*cm))

    # Summary stats
    win_rate = stats.get("win_rate", 0) * 100
    pf = stats.get("profit_factor", 0)
    total_pnl = stats.get("total_net_pnl", 0)
    summary_data = [
        ["Metric", "Value"],
        ["Total Trades", str(stats.get("total_trades", 0))],
        ["Win Rate", f"{win_rate:.1f}%"],
        ["Profit Factor", f"{pf:.2f}"],
        ["Net P&L", _format_inr(total_pnl)],
        ["Avg P&L / Trade", _format_inr(stats.get("avg_pnl_per_trade", 0))],
        ["Best Trade", _format_inr(stats.get("max_win", 0))],
        ["Worst Trade", _format_inr(stats.get("max_loss", 0))],
        ["Sharpe (approx)", f"{stats.get('sharpe_approx', 0):.2f}"],
    ]
    summary_table = Table(summary_data, colWidths=[7*cm, 7*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8f4fd")]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.5*cm))

    # Trade log table
    if trades:
        elements.append(Paragraph("Trade Log", styles["Heading2"]))
        trade_data = [["#", "Instrument", "Dir", "Entry", "Exit", "Qty", "P&L", "Exit Reason", "Score"]]
        for i, t in enumerate(trades, 1):
            pnl = t.net_pnl or 0
            pnl_str = _format_inr(pnl)
            trade_data.append([
                str(i),
                t.instrument[:20] if t.instrument else "",
                t.option_type or "",
                f"₹{t.entry_price:.2f}" if t.entry_price else "",
                f"₹{t.exit_price:.2f}" if t.exit_price else "OPEN",
                str(t.quantity or 0),
                pnl_str,
                t.exit_reason or "OPEN",
                f"{t.composite_score:.0f}" if t.composite_score else "",
            ])

        col_widths = [0.8*cm, 4.5*cm, 1*cm, 2.2*cm, 2.2*cm, 1.2*cm, 2.5*cm, 2.5*cm, 1.5*cm]
        trade_table = Table(trade_data, colWidths=col_widths)
        trade_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8ff")]),
        ]))
        elements.append(trade_table)

    doc.build(elements)
    logger.info("pdf_report_generated", path=str(filename))
    return filename


def generate_weekly_summary(week_end_date: date, mode: str = "paper") -> dict:
    """Aggregate stats for the past 7 calendar days, scoped to that exact window."""
    start = week_end_date - timedelta(days=7)
    trades = journal.get_trades_for_range(start, week_end_date, TradingMode(mode.upper()))
    stats = _compute_stats_for_trades(trades)
    return {
        "period": f"{start.strftime('%d %b')} – {week_end_date.strftime('%d %b %Y')}",
        "stats": stats,
        "generated_at": datetime.now(IST).isoformat(),
    }
