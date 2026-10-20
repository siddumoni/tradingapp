"""Historical and latest-candle data from Yahoo Finance, for development only.

This is an unofficial, best-effort data source. It does not provide exchange-grade
market depth, guaranteed real-time prices, or a service-level guarantee. No order
placement/modification/cancellation methods are implemented in this module.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.data_fetcher import DataFetchError, MarketClosedError, is_weekend

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
YAHOO_INTERVALS = {"5minute": "5m", "15minute": "15m", "day": "1d"}


class YahooFinanceMarketData:
    """Fetch OHLCV candles for NSE symbols using Yahoo Finance's ``.NS`` tickers."""

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None):
        self._ticker_factory = ticker_factory

    def _ticker(self, symbol: str) -> Any:
        if self._ticker_factory is not None:
            return self._ticker_factory(f"{symbol.upper()}.NS")
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataFetchError(
                "yfinance is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc
        return yf.Ticker(f"{symbol.upper()}.NS")

    @staticmethod
    def resolve_watchlist(symbols: list[str]) -> tuple[list[dict[str, str]], list[str]]:
        """Map configured NSE symbols to Yahoo Finance ticker symbols.

        Yahoo's symbol coverage is not authoritative; availability is confirmed when
        history is requested. This method therefore does not claim the ticker exists.
        """
        cleaned = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
        return [
            {"tradingsymbol": symbol, "yahoo_symbol": f"{symbol}.NS", "exchange": "NSE"}
            for symbol in cleaned
        ], []

    @staticmethod
    def _frame_to_candles(frame: Any) -> list[dict[str, Any]]:
        if frame is None or getattr(frame, "empty", True):
            return []
        candles: list[dict[str, Any]] = []
        for index, row in frame.iterrows():
            stamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
            if isinstance(stamp, datetime):
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=IST)
                else:
                    stamp = stamp.astimezone(IST)
            candles.append({
                "date": stamp,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if row.get("Volume") is not None else 0,
            })
        return candles

    def historical_candles(
        self,
        symbol: str,
        from_dt: datetime,
        to_dt: datetime,
        interval: str,
    ) -> list[dict[str, Any]]:
        if from_dt.tzinfo is None or to_dt.tzinfo is None:
            raise ValueError("Historical-data datetimes must be timezone-aware")
        if from_dt > to_dt:
            raise ValueError("from_dt must be <= to_dt")
        if interval not in YAHOO_INTERVALS:
            raise ValueError(f"Unsupported interval {interval!r}; choose one of {sorted(YAHOO_INTERVALS)}")
        ticker = self._ticker(symbol)
        try:
            frame = ticker.history(
                start=from_dt.astimezone(IST).replace(tzinfo=None),
                end=(to_dt.astimezone(IST) + timedelta(seconds=1)).replace(tzinfo=None),
                interval=YAHOO_INTERVALS[interval],
                auto_adjust=False,
                prepost=False,
                actions=False,
            )
        except Exception as exc:
            raise DataFetchError(
                f"Yahoo Finance history request failed for {symbol.upper()} ({interval}): {type(exc).__name__}: {exc}"
            ) from exc
        candles = self._frame_to_candles(frame)
        if not candles:
            LOGGER.warning("Yahoo Finance returned no %s candles for %s", interval, symbol.upper())
        return candles

    def fetch_symbol_bundle(self, symbol: str, now: datetime | None = None) -> dict[str, Any]:
        """Fetch recent 5/15-minute history and 60 calendar days of daily candles.

        Intraday history spans seven calendar days so indicator warm-up has enough
        completed bars early in the session. VWAP is grouped by session downstream,
        so including prior sessions does not mix their VWAP values. Yahoo data is
        still unofficial; ``quote`` is not a guaranteed live quote.
        """
        current = (now or datetime.now(IST)).astimezone(IST)
        if is_weekend(current.date()):
            raise MarketClosedError("Today is a weekend; no NSE cash-market session is expected.")
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        # Yahoo's intraday endpoint supports recent history, not unlimited history.
        # Seven calendar days covers several sessions even around a weekend/holiday.
        intraday_start = datetime.combine(current.date() - timedelta(days=7), dtime(9, 15), tzinfo=IST)
        daily_start = datetime.combine(current.date() - timedelta(days=60), dtime(0, 0), tzinfo=IST)
        intraday_5m = self.historical_candles(symbol, intraday_start, current, "5minute") if current.time() >= dtime(9, 15) else []
        intraday_15m = self.historical_candles(symbol, intraday_start, current, "15minute") if current.time() >= dtime(9, 15) else []
        daily = self.historical_candles(symbol, daily_start, current, "day")
        latest = intraday_5m[-1] if intraday_5m else None
        return {
            "symbol": symbol,
            "provider": "yahoo_finance_unofficial",
            "fetched_at": current.isoformat(),
            "quote": ({"last_price": latest["close"], "timestamp": latest["date"], "source": "latest_5m_candle_not_guaranteed_live"} if latest else None),
            "market_depth": None,
            "candles_5m": intraday_5m,
            "candles_15m": intraday_15m,
            "candles_daily_60d": daily,
            "data_quality_warning": "Unofficial Yahoo Finance data; prices may be delayed or incomplete. No market depth is available.",
        }


def symbols_from_watchlist_config(config: dict[str, Any]) -> list[str]:
    """Return unique symbols from the project's existing watchlist YAML mapping."""
    symbols = list(config.get("nifty50", [])) + list(config.get("additional_liquid_fo", []))
    return list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
