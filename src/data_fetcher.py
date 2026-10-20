"""Read-only Kite Connect market-data client for Phase 3.

This module intentionally contains no order placement/modification/cancellation calls.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import yaml

from src.auth import DEFAULT_TOKEN_CACHE, load_credentials, load_token_cache

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_WATCHLIST = PROJECT_ROOT / "config" / "watchlist.yaml"
INSTRUMENT_CACHE = PROJECT_ROOT / ".local" / "nse_instruments.json"


class DataFetchError(RuntimeError):
    """Raised when market data cannot be fetched or validated."""


class MarketClosedError(DataFetchError):
    """Raised when a data run is attempted on a weekend or outside market hours."""


@dataclass(frozen=True)
class DataClientConfig:
    retries: int = 3
    retry_delay_seconds: float = 1.0
    quote_batch_size: int = 200


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataFetchError(f"Could not read configuration file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataFetchError(f"Configuration file must contain a YAML mapping: {path}")
    return value


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def is_regular_market_hours(now: datetime) -> bool:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(IST)
    return (not is_weekend(local.date())) and dtime(9, 15) <= local.time().replace(tzinfo=None) <= dtime(15, 30)


def _retry(operation: Callable[[], Any], *, attempts: int, delay: float, label: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # Kite SDK has multiple exception classes for transient failures.
            last_error = exc
            if attempt == attempts:
                break
            wait = delay * attempt
            LOGGER.warning("%s failed (attempt %s/%s); retrying in %.1fs: %s", label, attempt, attempts, wait, type(exc).__name__)
            time.sleep(wait)
    raise DataFetchError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


class KiteMarketData: 
    """Read-only market-data wrapper around an authenticated KiteConnect client."""

    def __init__(self, kite: Any, config: DataClientConfig | None = None):
        self.kite = kite
        self.config = config or DataClientConfig()

    @classmethod
    def from_cached_session(cls, cache_path: Path = DEFAULT_TOKEN_CACHE, config: DataClientConfig | None = None) -> "KiteMarketData":
        api_key, _ = load_credentials()
        cached = load_token_cache(api_key, cache_path=cache_path)
        if cached is None:
            raise DataFetchError("No valid cached Kite token. Run: python -m src.main login")
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:
            raise DataFetchError("kiteconnect is missing. Run: python -m pip install -r requirements.txt") from exc
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(cached.access_token)
        return cls(kite, config=config)

    def instruments(self, cache_path: Path = INSTRUMENT_CACHE, now: datetime | None = None) -> list[dict[str, Any]]:
        """Return NSE instrument dump, caching one copy per IST calendar day."""
        current = (now or datetime.now(IST)).astimezone(IST)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("date") == current.date().isoformat() and isinstance(cached.get("instruments"), list):
                    return cached["instruments"]
            except (OSError, json.JSONDecodeError, AttributeError):
                LOGGER.warning("Instrument cache is invalid; downloading a fresh copy")
        rows = _retry(lambda: self.kite.instruments("NSE"), attempts=self.config.retries, delay=self.config.retry_delay_seconds, label="NSE instrument download")
        if not isinstance(rows, list) or not rows:
            raise DataFetchError("Kite returned an empty NSE instrument list")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temp_path.write_text(json.dumps({"date": current.date().isoformat(), "instruments": rows}, default=str), encoding="utf-8")
        temp_path.replace(cache_path)
        return rows

    @staticmethod
    def instrument_map(instruments: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(row.get("tradingsymbol", "")).upper(): row for row in instruments if row.get("tradingsymbol")}

    def resolve_watchlist(self, watchlist_path: Path = DEFAULT_WATCHLIST, instrument_rows: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
        config = load_yaml(watchlist_path)
        requested = list(dict.fromkeys([str(s).upper() for s in config.get("nifty50", []) + config.get("additional_liquid_fo", [])]))
        rows = instrument_rows if instrument_rows is not None else self.instruments()
        by_symbol = self.instrument_map(rows)
        resolved: list[dict[str, Any]] = []
        missing: list[str] = []
        for symbol in requested:
            row = by_symbol.get(symbol)
            if row and row.get("segment") == "NSE" and row.get("instrument_type") == "EQ":
                resolved.append(row)
            else:
                missing.append(symbol)
        return resolved, missing

    def ltp(self, symbols: Iterable[str]) -> dict[str, Any]:
        keys = [f"NSE:{symbol.upper()}" for symbol in dict.fromkeys(symbols)]
        if not keys:
            return {}
        result: dict[str, Any] = {}
        for offset in range(0, len(keys), self.config.quote_batch_size):
            batch = keys[offset:offset + self.config.quote_batch_size]
            data = _retry(lambda b=batch: self.kite.ltp(*b), attempts=self.config.retries, delay=self.config.retry_delay_seconds, label="LTP request")
            if isinstance(data, dict):
                result.update(data)
        return result

    def quotes(self, symbols: Iterable[str]) -> dict[str, Any]:
        keys = [f"NSE:{symbol.upper()}" for symbol in dict.fromkeys(symbols)]
        if not keys:
            return {}
        result: dict[str, Any] = {}
        for offset in range(0, len(keys), self.config.quote_batch_size):
            batch = keys[offset:offset + self.config.quote_batch_size]
            data = _retry(lambda b=batch: self.kite.quote(*b), attempts=self.config.retries, delay=self.config.retry_delay_seconds, label="Quote request")
            if isinstance(data, dict):
                result.update(data)
        return result

    def historical_candles(self, instrument_token: int, from_dt: datetime, to_dt: datetime, interval: str) -> list[dict[str, Any]]:
        if from_dt.tzinfo is None or to_dt.tzinfo is None:
            raise ValueError("Historical-data datetimes must be timezone-aware")
        if from_dt > to_dt:
            raise ValueError("from_dt must be <= to_dt")
        allowed = {"5minute", "15minute", "day"}
        if interval not in allowed:
            raise ValueError(f"Unsupported interval {interval!r}; choose one of {sorted(allowed)}")
        candles = _retry(lambda: self.kite.historical_data(instrument_token, from_dt.astimezone(IST), to_dt.astimezone(IST), interval), attempts=self.config.retries, delay=self.config.retry_delay_seconds, label=f"Historical {interval} candles")
        if candles is None:
            return []
        if not isinstance(candles, list):
            raise DataFetchError(f"Kite returned an unexpected response for {interval} candles")
        return candles

    def fetch_symbol_bundle(self, instrument: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
        """Fetch current quote and today's 5/15m plus 60 calendar days of daily candles."""
        current = (now or datetime.now(IST)).astimezone(IST)
        if is_weekend(current.date()):
            raise MarketClosedError("Today is a weekend; no NSE cash-market session is expected.")
        token = instrument.get("instrument_token")
        symbol = str(instrument.get("tradingsymbol", "")).upper()
        if not token or not symbol:
            raise DataFetchError("Instrument row must include instrument_token and tradingsymbol")
        quote_key = f"NSE:{symbol}"
        quote_data = self.quotes([symbol]).get(quote_key)
        start_today = datetime.combine(current.date(), dtime(9, 15), tzinfo=IST)
        daily_start = datetime.combine(current.date() - timedelta(days=60), dtime(0, 0), tzinfo=IST)
        return {
            "symbol": symbol,
            "instrument_token": int(token),
            "fetched_at": current.isoformat(),
            "quote": quote_data,
            "candles_5m": self.historical_candles(int(token), start_today, current, "5minute") if current.time() >= dtime(9, 15) else [],
            "candles_15m": self.historical_candles(int(token), start_today, current, "15minute") if current.time() >= dtime(9, 15) else [],
            "candles_daily_60d": self.historical_candles(int(token), daily_start, current, "day"),
        }


def market_session_message(now: datetime | None = None) -> str | None:
    """Return a graceful notice for weekends or outside normal NSE cash hours.

    Exchange holidays cannot be conclusively identified from weekday/time alone;
    empty historical candles should be treated as a no-data session, not a signal.
    """
    current = (now or datetime.now(IST)).astimezone(IST)
    if is_weekend(current.date()):
        return "NSE cash market is normally closed on weekends. No market-data scan was run."
    if current.time() < dtime(9, 15) or current.time() > dtime(15, 30):
        return "Outside normal NSE cash-market hours (09:15–15:30 IST). Historical daily data may still be available; intraday signals are not evaluated."
    return None
