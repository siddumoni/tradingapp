"""Manual Kite Connect login and local access-token cache.

This module intentionally exposes only authentication/session-validation
operations. It contains no order placement, modification, or cancellation code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_TOKEN_CACHE = PROJECT_ROOT / ".local" / "kite_token.json"


class AuthConfigurationError(RuntimeError):
    """Raised when required authentication configuration is missing/invalid."""


class TokenCacheError(RuntimeError):
    """Raised when a token cache cannot be read or written safely."""


@dataclass(frozen=True)
class CachedToken:
    access_token: str
    api_key: str
    created_at: datetime
    expires_at: datetime

    def is_valid_at(self, now: datetime) -> bool:
        """Return whether the cache timestamp says the token is not expired."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(IST) < self.expires_at.astimezone(IST)


def load_credentials(env_file: Path = ENV_FILE) -> tuple[str, str]:
    """Load Kite credentials from .env; never use hard-coded credentials."""
    # python-dotenv is used when installed. A tiny fallback keeps local unit tests
    # and diagnostics usable before dependencies have been installed. Existing
    # environment variables take precedence in either case.
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_file, override=False)
    except ImportError:
        if env_file.exists():
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    placeholders = {"", "replace_with_your_api_key", "replace_with_your_api_secret"}
    if api_key in placeholders or api_secret in placeholders:
        raise AuthConfigurationError(
            f"Set KITE_API_KEY and KITE_API_SECRET in {env_file}. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return api_key, api_secret


def token_expiry_for(now: datetime) -> datetime:
    """Return the next 06:00 IST boundary, when Kite tokens expire daily.

    Kite documents access tokens as expiring at 06:00 IST on the next day.
    This calculation is a conservative local cache cutoff; the API remains
    authoritative, so `check` also validates the session against Kite.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local_now = now.astimezone(IST)
    # Kite documents access-token expiry at 06:00 IST on the following day.
    # Always use the next calendar day's cutoff, even for a login before 06:00.
    expiry_date = local_now.date() + timedelta(days=1)
    return datetime.combine(expiry_date, time(6, 0), tzinfo=IST)


def save_token_cache(
    access_token: str,
    api_key: str,
    cache_path: Path = DEFAULT_TOKEN_CACHE,
    now: datetime | None = None,
) -> CachedToken:
    """Persist the token locally and restrict file permissions where supported."""
    current = (now or datetime.now(IST)).astimezone(IST)
    if not access_token.strip():
        raise TokenCacheError("Refusing to cache an empty access token")
    cached = CachedToken(
        access_token=access_token.strip(),
        api_key=api_key,
        created_at=current,
        expires_at=token_expiry_for(current),
    )
    payload = {
        "access_token": cached.access_token,
        "api_key": cached.api_key,
        "created_at": cached.created_at.isoformat(),
        "expires_at": cached.expires_at.isoformat(),
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            # Effective on POSIX; Windows access is primarily controlled by NTFS ACLs.
            os.chmod(cache_path, 0o600)
        except OSError:
            LOGGER.debug("Could not change token cache file permissions", exc_info=True)
    except OSError as exc:
        raise TokenCacheError(f"Could not write token cache at {cache_path}: {exc}") from exc
    return cached


def load_token_cache(
    api_key: str,
    cache_path: Path = DEFAULT_TOKEN_CACHE,
    now: datetime | None = None,
) -> CachedToken | None:
    """Load today's cached token, returning None if absent, mismatched or expired."""
    if not cache_path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = CachedToken(
            access_token=str(payload["access_token"]),
            api_key=str(payload["api_key"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            expires_at=datetime.fromisoformat(payload["expires_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise TokenCacheError(
            f"Token cache at {cache_path} is unreadable or malformed. "
            "Delete it and run the login command again."
        ) from exc

    current = (now or datetime.now(IST)).astimezone(IST)
    if cached.api_key != api_key or not cached.is_valid_at(current):
        return None
    return cached


def login(cache_path: Path = DEFAULT_TOKEN_CACHE) -> None:
    """Print login URL, accept request_token, exchange it, and cache the token."""
    api_key, api_secret = load_credentials()
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise AuthConfigurationError(
            "Dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc
    kite = KiteConnect(api_key=api_key)
    print("\nOpen this URL in your browser and complete Zerodha login:\n")
    print(kite.login_url())
    print(
        "\nAfter login, copy the `request_token` query parameter from your app's "
        "registered redirect URL. Do not paste the full URL."
    )
    request_token = input("request_token: ").strip()
    if not request_token:
        raise AuthConfigurationError("No request_token supplied; login cancelled.")

    try:
        session_data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:  # SDK exceptions vary by failure category.
        LOGGER.exception("Kite session generation failed")
        raise AuthConfigurationError(
            "Kite rejected the request_token. It may be expired or already used. "
            "Run login again and use a fresh request_token."
        ) from exc

    access_token = str(session_data.get("access_token", "")).strip()
    if not access_token:
        raise AuthConfigurationError("Kite response did not contain an access_token.")
    cached = save_token_cache(access_token, api_key, cache_path=cache_path)
    print("\nLogin succeeded. Access token saved locally (token value not displayed).")
    print(f"Cached token cutoff: {cached.expires_at.isoformat()}")
    print("Keep this computer account secure; the local token file is sensitive.")


def check_session(cache_path: Path = DEFAULT_TOKEN_CACHE) -> bool:
    """Validate the cached token with Kite's profile endpoint."""
    api_key, _ = load_credentials()
    cached = load_token_cache(api_key, cache_path=cache_path)
    if cached is None:
        print("No valid cached token. Run: python -m src.main login")
        return False

    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise AuthConfigurationError(
            "Dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(cached.access_token)
    try:
        profile = kite.profile()
    except Exception:
        LOGGER.exception("Kite session validation failed")
        print("Kite did not accept the cached session. Run the login command again.")
        return False

    # Avoid printing personal profile fields. A successful profile response is enough.
    if not isinstance(profile, dict) or not profile:
        print("Kite returned an unexpected profile response; session not confirmed.")
        return False
    print(f"Kite session is valid. Local cache cutoff: {cached.expires_at.isoformat()}")
    return True
