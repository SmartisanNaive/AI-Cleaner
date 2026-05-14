from __future__ import annotations

import base64
import os

from cryptography.fernet import Fernet, InvalidToken

from .constants import DATA_DIR

SECRET_FILE = DATA_DIR / "app.secret"


def _normalise_secret(raw: str) -> bytes:
    candidate = raw.encode("utf-8")
    try:
        base64.urlsafe_b64decode(candidate)
        if len(candidate) == 44:
            return candidate
    except Exception:
        pass
    return base64.urlsafe_b64encode(candidate.ljust(32, b"0")[:32])


def _load_or_create_secret() -> bytes:
    env_secret = os.getenv("AI_CLEANER_SECRET")
    if env_secret:
        return _normalise_secret(env_secret)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes().strip()
    secret = Fernet.generate_key()
    SECRET_FILE.write_bytes(secret)
    try:
        SECRET_FILE.chmod(0o600)
    except OSError:
        pass
    return secret


def get_fernet() -> Fernet:
    return Fernet(_load_or_create_secret())


def encrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    return get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None
