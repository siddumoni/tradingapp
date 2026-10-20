from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.data_fetcher import DataFetchError, MarketClosedError
from src.yahoo_data_fetcher import YahooFinanceMarketData, symbols_from_watchlist_config

IST = ZoneInfo("Asia/Kolkata")


class FakeTicker:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def history(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame


def sample_frame():
    index = pd.DatetimeIndex([datetime(2026, 10, 19, 9, 20, tzinfo=IST)])
    return pd.DataFrame(
        {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [5000]},
        index=index,
    )


def test_resolves_symbols_to_yahoo_tickers_without_claiming_validation():
    rows, missing = YahooFinanceMarketData.resolve_watchlist(["reliance", "TCS", "RELIANCE"])
    assert [row["yahoo_symbol"] for row in rows] == ["RELIANCE.NS", "TCS.NS"]
    assert missing == []


def test_historical_candles_converts_frame_and_uses_yahoo_interval():
    ticker = FakeTicker(sample_frame())
    client = YahooFinanceMarketData(ticker_factory=lambda symbol: ticker)
    start = datetime(2026, 10, 19, 9, 15, tzinfo=IST)
    candles = client.historical_candles("RELIANCE", start, start + timedelta(minutes=5), "5minute")
    assert candles[0]["close"] == 101.0
    assert candles[0]["volume"] == 5000
    assert ticker.calls[0]["interval"] == "5m"
    assert ticker.calls[0]["auto_adjust"] is False


def test_historical_candles_rejects_naive_datetime_and_unknown_interval():
    client = YahooFinanceMarketData(ticker_factory=lambda symbol: FakeTicker(sample_frame()))
    start = datetime(2026, 10, 19, 9, 15, tzinfo=IST)
    with pytest.raises(ValueError, match="timezone-aware"):
        client.historical_candles("RELIANCE", start.replace(tzinfo=None), start, "5minute")
    with pytest.raises(ValueError, match="Unsupported interval"):
        client.historical_candles("RELIANCE", start, start, "1minute")


def test_empty_history_returns_empty_list():
    client = YahooFinanceMarketData(ticker_factory=lambda symbol: FakeTicker(pd.DataFrame()))
    start = datetime(2026, 10, 19, 9, 15, tzinfo=IST)
    assert client.historical_candles("RELIANCE", start, start + timedelta(minutes=5), "5minute") == []


def test_provider_error_is_wrapped():
    class BrokenTicker:
        def history(self, **kwargs):
            raise RuntimeError("network down")
    client = YahooFinanceMarketData(ticker_factory=lambda symbol: BrokenTicker())
    start = datetime(2026, 10, 19, 9, 15, tzinfo=IST)
    with pytest.raises(DataFetchError, match="Yahoo Finance history request failed"):
        client.historical_candles("RELIANCE", start, start + timedelta(minutes=5), "5minute")


def test_weekends_are_rejected_for_bundle():
    client = YahooFinanceMarketData(ticker_factory=lambda symbol: FakeTicker(sample_frame()))
    saturday = datetime(2026, 10, 17, 10, 0, tzinfo=IST)
    with pytest.raises(MarketClosedError):
        client.fetch_symbol_bundle("RELIANCE", saturday)


def test_watchlist_config_symbols_are_deduplicated():
    assert symbols_from_watchlist_config({"nifty50": ["RELIANCE", "TCS"], "additional_liquid_fo": ["TCS", "INFY"]}) == ["RELIANCE", "TCS", "INFY"]
