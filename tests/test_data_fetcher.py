from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.data_fetcher import (
    DataClientConfig, DataFetchError, KiteMarketData, MarketClosedError,
    is_regular_market_hours, is_weekend, market_session_message,
)

IST = ZoneInfo("Asia/Kolkata")


class FakeKite:
    def __init__(self):
        self.instrument_calls = 0
        self.ltp_calls = []
        self.quote_calls = []
        self.historical_calls = []

    def instruments(self, exchange):
        self.instrument_calls += 1
        assert exchange == "NSE"
        return [{"tradingsymbol": "RELIANCE", "instrument_token": 1, "segment": "NSE", "instrument_type": "EQ"}]

    def ltp(self, *keys):
        self.ltp_calls.append(keys)
        return {key: {"last_price": 100.0} for key in keys}

    def quote(self, *keys):
        self.quote_calls.append(keys)
        return {key: {"last_price": 100.0, "depth": {"buy": [], "sell": []}} for key in keys}

    def historical_data(self, token, start, end, interval):
        self.historical_calls.append((token, start, end, interval))
        return [{"date": start, "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000}]


def test_weekend_and_market_hours():
    assert is_weekend(date(2026, 10, 17))  # Saturday
    assert not is_weekend(date(2026, 10, 19))  # Monday
    assert is_regular_market_hours(datetime(2026, 10, 19, 9, 15, tzinfo=IST))
    assert not is_regular_market_hours(datetime(2026, 10, 19, 8, 59, tzinfo=IST))
    assert market_session_message(datetime(2026, 10, 17, 10, 0, tzinfo=IST))
    assert market_session_message(datetime(2026, 10, 19, 8, 0, tzinfo=IST))
    assert market_session_message(datetime(2026, 10, 19, 10, 0, tzinfo=IST)) is None


def test_instrument_cache_is_reused_for_same_day(tmp_path: Path):
    kite = FakeKite()
    client = KiteMarketData(kite, DataClientConfig(retries=1, retry_delay_seconds=0))
    cache = tmp_path / "instruments.json"
    now = datetime(2026, 10, 19, 10, 0, tzinfo=IST)
    assert len(client.instruments(cache, now)) == 1
    assert len(client.instruments(cache, now)) == 1
    assert kite.instrument_calls == 1


def test_instrument_cache_refreshes_on_new_day(tmp_path: Path):
    kite = FakeKite()
    client = KiteMarketData(kite, DataClientConfig(retries=1, retry_delay_seconds=0))
    cache = tmp_path / "instruments.json"
    client.instruments(cache, datetime(2026, 10, 19, 10, 0, tzinfo=IST))
    client.instruments(cache, datetime(2026, 10, 20, 10, 0, tzinfo=IST))
    assert kite.instrument_calls == 2


def test_resolve_watchlist_reports_missing_symbols(tmp_path: Path):
    kite = FakeKite()
    client = KiteMarketData(kite, DataClientConfig(retries=1, retry_delay_seconds=0))
    watchlist = tmp_path / "watchlist.yaml"
    watchlist.write_text("nifty50:\n  - RELIANCE\n  - MISSING\nadditional_liquid_fo: []\n", encoding="utf-8")
    rows, missing = client.resolve_watchlist(watchlist, FakeKite().instruments("NSE"))
    assert [row["tradingsymbol"] for row in rows] == ["RELIANCE"]
    assert missing == ["MISSING"]


def test_ltp_and_quote_batching():
    kite = FakeKite()
    client = KiteMarketData(kite, DataClientConfig(retries=1, retry_delay_seconds=0, quote_batch_size=2))
    symbols = ["RELIANCE", "TCS", "INFY"]
    assert len(client.ltp(symbols)) == 3
    assert len(kite.ltp_calls) == 2
    assert len(client.quotes(symbols)) == 3
    assert len(kite.quote_calls) == 2


def test_historical_candles_validates_interval_and_timezone():
    client = KiteMarketData(FakeKite(), DataClientConfig(retries=1, retry_delay_seconds=0))
    start = datetime(2026, 10, 19, 9, 15, tzinfo=IST)
    end = start + timedelta(minutes=15)
    assert client.historical_candles(1, start, end, "5minute")
    with pytest.raises(ValueError):
        client.historical_candles(1, start.replace(tzinfo=None), end, "5minute")
    with pytest.raises(ValueError):
        client.historical_candles(1, start, end, "1minute")


def test_retry_then_success(monkeypatch):
    client = KiteMarketData(FakeKite(), DataClientConfig(retries=2, retry_delay_seconds=0))
    attempts = {"n": 0}
    def flaky(*keys):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("temporary")
        return {key: {"last_price": 1} for key in keys}
    client.kite.ltp = flaky
    assert client.ltp(["RELIANCE"])
    assert attempts["n"] == 2


def test_retries_exhausted_raises_clear_error():
    kite = FakeKite()
    def fail(*keys):
        raise RuntimeError("down")
    kite.ltp = fail
    client = KiteMarketData(kite, DataClientConfig(retries=2, retry_delay_seconds=0))
    with pytest.raises(DataFetchError, match="LTP request failed after 2 attempts"):
        client.ltp(["RELIANCE"])
