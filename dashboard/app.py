"""
FastAPI web dashboard — JWT-authenticated, real-time P&L monitoring.
Provides: live positions, controls (pause/resume/stop), mode switching,
trade journal, and performance analytics.
"""

from datetime import datetime, timedelta
from typing import Optional
import yaml
import structlog
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import jwt as pyjwt

from core.security.vault import vault
from core.security.pin_auth import verify_pin
from core.execution.risk_manager import risk_manager
from database.trade_journal import journal
from database.models import TradingMode

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))

app = FastAPI(title="NiftySniper Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

SECRET_KEY = vault.dashboard_secret_key
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = _cfg["security"]["jwt_expiry_hours"]

_system_mode = _cfg["system"]["mode"]   # "paper" | "live"
_bot_paused = False


# ── JWT Helpers ────────────────────────────────────────────────────────────────

def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGO)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
        return payload.get("sub")
    except Exception:
        return None


def get_current_user(request: Request) -> Optional[str]:
    token = request.cookies.get("access_token")
    if not token:
        return None
    return decode_token(token)


def require_auth(request: Request) -> str:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── Auth Routes ────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    pin: str = Form(...),
):
    admin_user = _cfg["dashboard"].get("username", "admin")
    # Always run verify_pin (bcrypt), regardless of username match, so response
    # timing can't be used to enumerate whether a username is valid.
    username_ok = username == admin_user
    pin_ok = verify_pin(pin)
    if not (username_ok and pin_ok):
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid credentials"})
    token = create_token(username)
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        "access_token", token, httponly=True, samesite="strict",
        max_age=JWT_EXPIRE_HOURS * 3600,
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response


# ── Dashboard Home ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    stats = risk_manager.get_daily_stats()
    perf = journal.get_performance_stats(mode=TradingMode(_system_mode.upper()))
    trades_today = journal.get_trades_today(TradingMode(_system_mode.upper()))
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user,
        "mode": _system_mode,
        "paused": _bot_paused,
        "daily_stats": stats,
        "performance": perf,
        "trades_today": trades_today,
        "generated_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
    })


# ── API: Live Data ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status(request: Request):
    require_auth(request)
    return {
        "mode": _system_mode,
        "paused": _bot_paused,
        "halted": risk_manager.is_halted,
        "daily_stats": risk_manager.get_daily_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/positions")
async def api_positions(request: Request):
    require_auth(request)
    open_trades = journal.get_open_trades(TradingMode(_system_mode.upper()))
    return [{
        "trade_id": t.trade_id,
        "instrument": t.instrument,
        "direction": t.option_type,
        "entry_price": t.entry_price,
        "quantity": t.quantity,
        "stop_loss": t.stop_loss,
        "tp1": t.tp1, "tp2": t.tp2, "tp3": t.tp3,
        "regime": t.regime,
        "score": t.composite_score,
        "mode": t.mode.value if t.mode else _system_mode,
    } for t in open_trades]


@app.get("/api/trades")
async def api_trades(request: Request, limit: int = 50):
    require_auth(request)
    trades = journal.get_recent_trades(n=limit, mode=TradingMode(_system_mode.upper()))
    return [{
        "trade_id": t.trade_id,
        "instrument": t.instrument,
        "direction": t.option_type,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "net_pnl": t.net_pnl,
        "exit_reason": t.exit_reason,
        "status": t.status.value if t.status else "",
        "created_at": t.created_at.isoformat() if t.created_at else "",
        "score": t.composite_score,
        "confluence": t.confluence_count,
    } for t in trades]


@app.get("/api/performance")
async def api_performance(request: Request, mode: str = None):
    require_auth(request)
    m = TradingMode((mode or _system_mode).upper())
    stats = journal.get_performance_stats(mode=m)
    from core.learning.performance_tracker import performance_tracker
    readiness = performance_tracker.check_live_escalation_readiness()
    from core.learning.adaptive_weights import get_weight_history
    return {
        "stats": stats,
        "live_readiness": readiness,
        "weight_history": get_weight_history(10),
    }


# ── API: Controls ──────────────────────────────────────────────────────────────

@app.post("/api/pause")
async def api_pause(request: Request):
    require_auth(request)
    global _bot_paused
    _bot_paused = True
    journal.log_event("PAUSE", "Trading paused via dashboard")
    return {"status": "paused"}


@app.post("/api/resume")
async def api_resume(request: Request):
    require_auth(request)
    global _bot_paused
    _bot_paused = False
    risk_manager.resume()
    journal.log_event("RESUME", "Trading resumed via dashboard")
    return {"status": "running"}


@app.post("/api/emergency_stop")
async def api_emergency_stop(request: Request):
    require_auth(request)
    from core.execution.paper_trader import paper_trader
    from core.execution.order_manager import order_manager
    from core.execution.position_manager import position_manager
    risk_manager.halt("Emergency stop via dashboard")
    prices = position_manager.get_current_prices_for_mode(_system_mode)
    if _system_mode == "paper":
        paper_trader.force_squareoff_all(prices)
    else:
        position_manager.squareoff_all(prices, mode="live")
        order_manager.cancel_all_pending()
    journal.log_event("EMERGENCY_STOP", "Emergency stop triggered via dashboard", severity="CRITICAL")
    return {"status": "stopped", "message": "All positions squared off"}


@app.post("/api/set_mode")
async def api_set_mode(request: Request, mode: str = Form(...), pin: str = Form(...)):
    require_auth(request)
    global _system_mode
    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="Mode must be 'paper' or 'live'")
    if mode == "live":
        if not verify_pin(pin):
            raise HTTPException(status_code=403, detail="Invalid PIN for live mode activation")
        # Verify escalation criteria
        from core.learning.performance_tracker import performance_tracker
        readiness = performance_tracker.check_live_escalation_readiness()
        if not readiness["ready_for_live"]:
            raise HTTPException(status_code=403, detail=f"Escalation criteria not met: {readiness}")
    _system_mode = mode
    journal.log_event("MODE_CHANGE", f"Trading mode changed to {mode}", severity="WARNING")
    return {"status": "ok", "mode": mode}


@app.post("/api/cancel_trade/{trade_id}")
async def api_cancel_trade(trade_id: str, request: Request):
    require_auth(request)
    trade = journal.get_trade(trade_id)
    if not trade:
        return {"status": "not_found", "trade_id": trade_id}

    from core.market_data.kite_client import kite_client
    try:
        price = kite_client.get_ltp([f"{trade.exchange}:{trade.instrument}"]).get(
            f"{trade.exchange}:{trade.instrument}", trade.entry_price
        )
    except Exception:
        price = trade.entry_price

    if _system_mode == "live":
        from core.execution.order_manager import order_manager
        remaining = trade.remaining_quantity if trade.remaining_quantity is not None else trade.quantity
        order_manager.close_position(trade_id, trade.instrument, trade.exchange, remaining, "MANUAL", price)
    else:
        from core.execution.paper_trader import paper_trader
        paper_trader.force_squareoff(trade_id, price)
    return {"status": "cancelled", "trade_id": trade_id}


@app.get("/api/report/{date_str}")
async def api_download_report(date_str: str, request: Request):
    require_auth(request)
    from datetime import date
    from notifications.report_generator import generate_daily_csv
    try:
        trading_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    path = generate_daily_csv(trading_date, _system_mode)
    return FileResponse(str(path), filename=path.name, media_type="text/csv")
