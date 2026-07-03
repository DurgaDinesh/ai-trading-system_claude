"""Full signal pipeline diagnostic — run this to see exactly why trades aren't firing."""

import yaml
import pytz
from datetime import datetime

cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))
IST = pytz.timezone("Asia/Kolkata")
now = datetime.now(IST)

print("\n" + "="*60)
print("  NiftySniper Signal Diagnostic")
print("="*60)

# ── Gate 1: Time ──────────────────────────────────────────────
print("\n[GATE 1] Session Time")
ist_str = now.strftime("%H:%M")
weekday = now.weekday() + 1  # Mon=1
allowed_days = cfg["session"]["allowed_trading_days"]
print(f"  Current IST : {now.strftime('%H:%M:%S %A')}")
print(f"  Weekday     : {weekday} (allowed: {allowed_days})")
day_ok = weekday in allowed_days
time_ok = "09:30" <= ist_str <= "14:30"
print(f"  Day OK?     : {'YES' if day_ok else 'NO - not a trading day'}")
print(f"  Time OK?    : {'YES (within 9:30-14:30)' if time_ok else 'NO - outside trading hours'}")

# ── Gate 2: Fetch data ────────────────────────────────────────
print("\n[GATE 2] Market Data")
try:
    from core.market_data.historical import fetch_historical_yfinance
    df = fetch_historical_yfinance("NIFTY", period="10d", interval="5m")
    print(f"  Nifty 5m bars fetched: {len(df)}")
    if not df.empty:
        print(f"  Last close : {df['close'].iloc[-1]:.2f}")
        print(f"  Data OK?   : {'YES' if len(df) >= 50 else 'NO - need at least 50 bars'}")
    else:
        print("  Data OK?   : NO - empty dataframe")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Gate 3: Technical indicators ──────────────────────────────
print("\n[GATE 3] Technical Indicators")
try:
    from core.analysis.technical import (
        compute_all, get_ema_stack_signal, get_rsi_signal,
        get_macd_signal, get_vwap_signal, get_supertrend_signal
    )
    df = compute_all(df)
    row = df.iloc[-1]
    ema  = get_ema_stack_signal(df)
    rsi  = get_rsi_signal(df)
    macd = get_macd_signal(df)
    vwap = get_vwap_signal(df)
    st   = get_supertrend_signal(df)
    label = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    print(f"  EMA Stack   : {label[ema]}")
    print(f"  RSI         : {label[rsi]}  (value={row.get('rsi', 'N/A'):.1f})" if 'rsi' in df.columns else f"  RSI: {label[rsi]}")
    print(f"  MACD        : {label[macd]}")
    print(f"  VWAP        : {label[vwap]}  (close={row.get('close',0):.1f} vs vwap={row.get('vwap',0):.1f})" if 'vwap' in df.columns else f"  VWAP: {label[vwap]}")
    print(f"  Supertrend  : {label[st]}")
    signals = [ema, rsi, macd, vwap, st]
    bullish = signals.count(1)
    bearish = signals.count(-1)
    print(f"  Votes       : {bullish} bullish, {bearish} bearish")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Gate 4: Regime ────────────────────────────────────────────
print("\n[GATE 4] Regime Detection")
try:
    from core.signals.regime_detector import detect_regime
    global_ctx = {}
    news_ctx = {}
    options_ctx = {}
    regime = detect_regime(df, global_ctx, news_ctx, options_ctx)
    print(f"  Regime      : {regime.regime.value}")
    print(f"  Direction   : {regime.direction} ({'+1=Bullish' if regime.direction==1 else '-1=Bearish' if regime.direction==-1 else '0=Neutral'})")
    print(f"  Confidence  : {regime.confidence:.0%}")
    tradeable = regime.regime.value not in ("UNCERTAIN","HIGH_VOLATILITY","SIDEWAYS_LOW_VOL")
    print(f"  Tradeable?  : {'YES' if tradeable else 'NO - regime blocked'}")
    print(f"  Rationale   : {'; '.join(regime.rationale)}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Gate 5: Composite Score ───────────────────────────────────
print("\n[GATE 5] Composite Score & Confluence")
try:
    from core.signals.signal_engine import compute_composite_score
    score, triggered, confluence = compute_composite_score(df, regime, options_ctx, global_ctx, news_ctx)
    min_conf  = cfg["signals"]["min_confluence"]
    min_score = cfg["signals"]["min_composite_score"]
    print(f"  Score       : {score:.1f} / 100  (need >= {min_score})")
    print(f"  Confluence  : {confluence} indicators  (need >= {min_conf})")
    print(f"  Triggered   : {triggered}")
    print(f"  Score OK?   : {'YES' if score >= min_score else 'NO'}")
    print(f"  Confluence? : {'YES' if confluence >= min_conf else 'NO'}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Gate 6: R:R Ratio ─────────────────────────────────────────
# SL/TP are computed on the actual option premium once a strike is picked
# (strategy_selector.resolve_tradeable_instrument), not on the spot index.
# Here we can only validate the configured ratio, not a live premium.
print("\n[GATE 6] Risk:Reward Check (configured premium percentages)")
try:
    sl_pct = cfg["signals"]["sl_premium_pct"]
    tp1_pct = cfg["signals"]["tp_premium_pct"]["tp1"]
    rr = tp1_pct / sl_pct
    min_rr = cfg["signals"]["min_rr_ratio"]
    print(f"  SL          : -{sl_pct:.0%} of entry premium")
    print(f"  TP1         : +{tp1_pct:.0%} of entry premium")
    print(f"  R:R         : {rr:.2f}  (need >= {min_rr})")
    print(f"  R:R OK?     : {'YES' if rr >= min_rr else 'NO'}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60)
print("  VERDICT")
print("="*60)
try:
    all_ok = day_ok and time_ok and tradeable and score >= min_score and confluence >= min_conf and rr >= min_rr
    if all_ok:
        print("  ALL GATES PASS — a trade WOULD fire right now")
    else:
        fails = []
        if not day_ok:   fails.append("Not a trading day")
        if not time_ok:  fails.append("Outside 9:30-14:30 IST")
        if not tradeable: fails.append(f"Regime={regime.regime.value} (not tradeable)")
        if confluence < min_conf: fails.append(f"Confluence={confluence} < {min_conf}")
        if score < min_score: fails.append(f"Score={score:.1f} < {min_score}")
        if rr < min_rr: fails.append(f"R:R={rr:.2f} < {min_rr}")
        print("  BLOCKED by:")
        for f in fails:
            print(f"    - {f}")
except Exception as e:
    print(f"  ERROR computing verdict: {e}")
print("="*60 + "\n")
