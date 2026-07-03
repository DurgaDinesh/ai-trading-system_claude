"""6-digit PIN authentication using bcrypt hashing."""

import time
import threading
import bcrypt
from core.security.vault import vault

_state_lock = threading.Lock()
_failed_attempts = 0
_locked_until = 0.0

MAX_ATTEMPTS = 5          # Consecutive failures before lockout
LOCKOUT_SECONDS = 300     # 5-minute lockout — this gates live-order/dashboard access


def verify_pin(plain_pin: str) -> bool:
    """
    Return True if plain_pin matches the stored bcrypt hash.
    Locks out ALL PIN verification (not just the current caller) for
    LOCKOUT_SECONDS after MAX_ATTEMPTS consecutive failures, so this can't
    be brute-forced across the 6-digit space — bcrypt cost alone isn't a
    meaningful control for software gating live capital deployment.
    """
    global _failed_attempts, _locked_until

    with _state_lock:
        if time.time() < _locked_until:
            return False

    valid_format = bool(plain_pin) and plain_pin.isdigit() and len(plain_pin) == 6
    ok = False
    if valid_format:
        stored_hash = vault.trade_pin_hash.encode()
        ok = bcrypt.checkpw(plain_pin.encode(), stored_hash)

    with _state_lock:
        if ok:
            _failed_attempts = 0
        else:
            _failed_attempts += 1
            if _failed_attempts >= MAX_ATTEMPTS:
                _locked_until = time.time() + LOCKOUT_SECONDS
                _failed_attempts = 0
    return ok


def generate_pin_hash(plain_pin: str) -> str:
    """Utility to generate a bcrypt hash for a new PIN (run once during setup)."""
    if not plain_pin.isdigit() or len(plain_pin) != 6:
        raise ValueError("PIN must be exactly 6 digits.")
    return bcrypt.hashpw(plain_pin.encode(), bcrypt.gensalt()).decode()


def require_pin_for_live(func):
    """Decorator that prompts PIN verification before executing live orders."""
    def wrapper(*args, **kwargs):
        import getpass
        pin = getpass.getpass("Enter 6-digit trading PIN: ")
        if not verify_pin(pin):
            raise PermissionError("Invalid PIN. Live order blocked.")
        return func(*args, **kwargs)
    return wrapper
