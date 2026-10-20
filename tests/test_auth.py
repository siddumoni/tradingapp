from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.auth import (
    AuthConfigurationError,
    load_credentials,
    load_token_cache,
    save_token_cache,
    token_expiry_for,
)

IST = ZoneInfo("Asia/Kolkata")


def test_expiry_before_six_am_is_next_day() -> None:
    now = datetime(2026, 10, 17, 5, 30, tzinfo=IST)
    assert token_expiry_for(now) == datetime(2026, 10, 18, 6, 0, tzinfo=IST)


def test_expiry_after_six_am_is_next_day() -> None:
    now = datetime(2026, 10, 17, 9, 0, tzinfo=IST)
    assert token_expiry_for(now) == datetime(2026, 10, 18, 6, 0, tzinfo=IST)


def test_expiry_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        token_expiry_for(datetime(2026, 10, 17, 9, 0))


def test_save_and_load_token_cache(tmp_path: Path) -> None:
    cache = tmp_path / "token.json"
    now = datetime(2026, 10, 17, 9, 0, tzinfo=IST)
    save_token_cache("dummy-token", "dummy-key", cache_path=cache, now=now)

    loaded = load_token_cache(
        "dummy-key", cache_path=cache, now=datetime(2026, 10, 17, 10, 0, tzinfo=IST)
    )
    assert loaded is not None
    assert loaded.access_token == "dummy-token"
    assert loaded.expires_at == datetime(2026, 10, 18, 6, 0, tzinfo=IST)


def test_expired_token_cache_is_ignored(tmp_path: Path) -> None:
    cache = tmp_path / "token.json"
    save_token_cache(
        "dummy-token", "dummy-key", cache_path=cache,
        now=datetime(2026, 10, 17, 9, 0, tzinfo=IST),
    )
    loaded = load_token_cache(
        "dummy-key", cache_path=cache, now=datetime(2026, 10, 18, 6, 0, tzinfo=IST)
    )
    assert loaded is None


def test_cache_for_different_api_key_is_ignored(tmp_path: Path) -> None:
    cache = tmp_path / "token.json"
    save_token_cache(
        "dummy-token", "key-a", cache_path=cache,
        now=datetime(2026, 10, 17, 9, 0, tzinfo=IST),
    )
    assert load_token_cache(
        "key-b", cache_path=cache, now=datetime(2026, 10, 17, 10, 0, tzinfo=IST)
    ) is None


def test_malformed_cache_raises_clear_error(tmp_path: Path) -> None:
    cache = tmp_path / "token.json"
    cache.write_text("not-json", encoding="utf-8")
    from src.auth import TokenCacheError
    with pytest.raises(TokenCacheError, match="unreadable or malformed"):
        load_token_cache("key", cache_path=cache)


def test_credentials_require_non_placeholder_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "KITE_API_KEY=replace_with_your_api_key\nKITE_API_SECRET=replace_with_your_api_secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_API_SECRET", raising=False)
    with pytest.raises(AuthConfigurationError, match="Set KITE_API_KEY"):
        load_credentials(env_file)
