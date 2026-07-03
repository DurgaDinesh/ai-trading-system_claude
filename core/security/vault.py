"""AES-256 encrypted credential vault using Fernet symmetric encryption."""

import os
import json
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")


def _get_fernet() -> Fernet:
    key = os.environ.get("VAULT_ENCRYPTION_KEY")
    if not key:
        raise EnvironmentError(
            "VAULT_ENCRYPTION_KEY not set. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


def encrypt_secret(plain_text: str) -> str:
    """Encrypt a plaintext secret and return base64-encoded ciphertext."""
    return _get_fernet().encrypt(plain_text.encode()).decode()


def decrypt_secret(cipher_text: str) -> str:
    """Decrypt a Fernet-encrypted ciphertext."""
    try:
        return _get_fernet().decrypt(cipher_text.encode()).decode()
    except InvalidToken:
        raise ValueError("Decryption failed — invalid key or corrupted ciphertext.")


def get_credential(env_var: str) -> str:
    """
    Retrieve a credential from the environment.
    Supports both raw values and Fernet-encrypted values (prefixed with 'enc:').
    """
    value = os.environ.get(env_var)
    if not value:
        raise EnvironmentError(f"Missing environment variable: {env_var}")
    if value.startswith("enc:"):
        return decrypt_secret(value[4:])
    return value


class Vault:
    """Manages loading and accessing all API credentials securely."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def _load(self, key: str, env_var: str) -> str:
        if key not in self._cache:
            self._cache[key] = get_credential(env_var)
        return self._cache[key]

    @property
    def kite_api_key(self) -> str:
        return self._load("kite_api_key", "KITE_API_KEY")

    @property
    def kite_api_secret(self) -> str:
        return self._load("kite_api_secret", "KITE_API_SECRET")

    @property
    def kite_user_id(self) -> str:
        return self._load("kite_user_id", "KITE_USER_ID")

    @property
    def anthropic_api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def news_api_key(self) -> str:
        return self._load("news_api_key", "NEWS_API_KEY")

    @property
    def telegram_bot_token(self) -> str:
        return self._load("telegram_bot_token", "TELEGRAM_BOT_TOKEN")

    @property
    def telegram_chat_id(self) -> str:
        return self._load("telegram_chat_id", "TELEGRAM_CHAT_ID")

    @property
    def telegram_admin_chat_id(self) -> str:
        return self._load("telegram_admin_chat_id", "TELEGRAM_ADMIN_CHAT_ID")

    @property
    def dashboard_secret_key(self) -> str:
        return self._load("dashboard_secret_key", "DASHBOARD_SECRET_KEY")

    @property
    def trade_pin_hash(self) -> str:
        return self._load("trade_pin_hash", "TRADE_PIN_HASH")


vault = Vault()
