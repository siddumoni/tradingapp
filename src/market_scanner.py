"""Phase 6 market scanner and fail-closed advisory candidate builder.

This module is provider-agnostic. It consumes normalized OHLCV bundles from the
existing Kite/Yahoo adapters, evaluates completed candles only, and never places
or modifies orders. Yahoo data is explicitly treated as unverified for live use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time as dtime, timedelta
from html import unescape
import re
from urllib.parse import unquote
from math import isfinite
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from src.risk_engine import RiskSettings, TradeAssessment, assess_trade
from src.cost_calculator import ChargeRates
from src.strategies.market_regime import classify_market_regime
from src.strategies.opening_range import opening_range_signal
from src.strategies.signal_planner import plan_atr_levels
from src.strategies.vwap_pullback import vwap_pullback_signal
from src.strategies.vwap_trend import vwap_trend_signal

IST = ZoneInfo("Asia/Kolkata")
OHLCV = ["open", "high", "low", "close", "volume"]
STRATEGIES = ("opening_range", "vwap_trend", "vwap_pullback")


@dataclass(frozen=True)
class ScanCandidate:
    symbol: str
    action: str
    side: str | None
    strategy: str | None
    confidence: int
    market_regime: str
    signal_time: str | None
    reference_entry: float | None
    stop_loss: float | None
    target_1: float | None
    target_2: float | None
    quantity: int
    estimated_net_risk_inr: float | None
    estimated_net_reward_t1_inr: float | None
    net_rr_t1: float | None
    margin_verified: bool
    data_verified_live: bool
    news_status: str
    news_risk: str  # Backward-compatible aggregate risk.
    reasons: list[str]
    market_context_status: str = "UNKNOWN"
    market_context_risk: str = "UNKNOWN"
    company_headline_match_status: str = "UNKNOWN"
    company_specific_risk: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candles_to_frame(candles: Iterable[dict[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    """Normalize provider candles to an indexed OHLCV frame."""
    if candles is None:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(candles, pd.DataFrame):
        frame = candles.copy()
        if "date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).set_index("date")
    else:
        rows = list(candles)
        if not rows:
            return pd.DataFrame(columns=OHLCV)
        frame = pd.DataFrame(rows)
        time_col = next((c for c in ("date", "timestamp", "time") if c in frame.columns), None)
        if time_col is None:
            raise ValueError("Candles must include a date, timestamp, or time field")
        frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[time_col]).set_index(time_col)
    rename = {c: c.lower() for c in frame.columns if c.lower() in OHLCV}
    frame = frame.rename(columns=rename)
    missing = set(OHLCV).difference(frame.columns)
    if missing:
        raise ValueError(f"Candles are missing OHLCV columns: {sorted(missing)}")
    frame = frame[OHLCV].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(IST)
    else:
        frame.index = frame.index.tz_convert(IST)
    for col in OHLCV:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    if frame.empty:
        return frame
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    return frame[~frame.index.duplicated(keep="last")]


def add_scanner_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute the initial indicator set using pandas only (no external TA library)."""
    missing = set(OHLCV).difference(candles.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    df = candles.copy().sort_index()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    df["ema_9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False, min_periods=21).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.where(loss != 0, 100.0)
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    typical = (high + low + close) / 3.0
    session = pd.Index(df.index.date, name="session")
    cumulative_pv = (typical * volume).groupby(session).cumsum()
    cumulative_volume = volume.groupby(session).cumsum().replace(0, float("nan"))
    df["vwap"] = cumulative_pv / cumulative_volume
    df["volume_sma_20"] = volume.rolling(20, min_periods=5).mean()
    return df


def completed_candles(frame: pd.DataFrame, interval_minutes: int, now: datetime) -> pd.DataFrame:
    """Drop any candle whose full interval has not ended by `now`."""
    if frame.empty:
        return frame.copy()
    current = now.replace(tzinfo=IST) if now.tzinfo is None else now.astimezone(IST)
    ends = frame.index + pd.to_timedelta(interval_minutes, unit="m")
    return frame.loc[ends <= current].copy()


def _is_market_session(now: datetime) -> bool:
    current = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    return current.weekday() < 5 and dtime(9, 15) <= current.time() <= dtime(15, 30)


def _stock_15m_regime(frame: pd.DataFrame) -> str:
    """Lightweight multi-timeframe confirmation using the last three 15m closes."""
    if frame.empty or len(frame) < 3 or not {"close", "vwap"}.issubset(frame.columns):
        return "UNKNOWN"
    tail = frame.tail(3)
    close_values = tail["close"].astype(float).tolist()
    vwap = float(tail.iloc[-1]["vwap"])
    if pd.isna(vwap):
        return "UNKNOWN"
    rising = close_values[0] < close_values[1] < close_values[2]
    falling = close_values[0] > close_values[1] > close_values[2]
    last_close = close_values[-1]
    if rising and last_close > vwap:
        return "BULLISH"
    if falling and last_close < vwap:
        return "BEARISH"
    return "CHOPPY"


def _latest_candle_age_minutes(frame: pd.DataFrame, now: datetime, interval_minutes: int) -> float | None:
    if frame.empty:
        return None
    current = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    end = frame.index[-1] + timedelta(minutes=interval_minutes)
    return max(0.0, (current - end.to_pydatetime()).total_seconds() / 60.0)


def _clean_news_text(value: Any) -> str:
    """Remove RSS HTML/encoded URLs so markup cannot trigger risk keywords."""
    text = unquote(unescape(str(value or "")))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _has_term(text: str, term: str) -> bool:
    """Match phrases/keywords without accidental substring matches."""
    escaped = re.escape(term.casefold()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text.casefold()) is not None


def _news_risk_breakdown(
    symbol: str, news_items: list[dict[str, Any]]
) -> tuple[str, str, str, str, str, str, list[str]]:
    """Separate market context from company-specific headline matches.

    No match means no match in the supplied items, not proof that no news exists.
    This keyword triage is not sentiment analysis or verification.
    """
    if not news_items:
        reason = "No news headlines were available; news coverage is unavailable."
        return "UNAVAILABLE", "UNKNOWN", "UNAVAILABLE", "UNKNOWN", "UNAVAILABLE", "UNKNOWN", [reason]

    symbol = symbol.strip().upper()
    aliases = {
        "RELIANCE": ("Reliance Industries", "Reliance"),
        "HDFCBANK": ("HDFC Bank", "HDFC"),
        "ICICIBANK": ("ICICI Bank", "ICICI"),
        "SBIN": ("State Bank of India", "SBI"),
        "INFY": ("Infosys",),
    }
    symbol_terms = {symbol, symbol.replace("&", " "), *aliases.get(symbol, ())}
    market_terms = (
        "NIFTY", "SENSEX", "INDIAN STOCK MARKET", "RBI", "SEBI", "RUPEE",
        "CRUDE OIL", "GIFT NIFTY", "STOCK MARKET", "INDIAN EQUITIES",
    )
    high_terms = (
        "fraud", "raid", "default", "resigns", "resignation", "ban", "probe",
        "investigation", "downgrade", "profit warning", "earnings warning",
        "regulatory action", "penalty", "results below", "guidance cut",
        "cyberattack", "insolvency", "trading suspended",
    )
    medium_terms = (
        "results", "earnings", "merger", "acquisition", "order win", "large order",
        "policy", "rate decision", "inflation", "budget", "strike", "shutdown",
        "court", "approval", "rejection", "guidance", "crude oil", "currency",
        "market falls", "market rallies",
    )
    company_matches: list[tuple[str, str]] = []
    market_matches: list[tuple[str, str]] = []

    for item in news_items:
        title = _clean_news_text(item.get("title", ""))
        summary = _clean_news_text(item.get("summary", ""))
        if not title:
            continue
        searchable = f"{title} {summary}"
        is_company = any(_has_term(searchable, term) for term in symbol_terms)
        is_market = any(_has_term(searchable, term) for term in market_terms)
        if not (is_company or is_market):
            continue

        title_lower = title.casefold()
        severity = (
            "HIGH" if any(_has_term(title_lower, term) for term in high_terms)
            else "MEDIUM" if any(_has_term(title_lower, term) for term in medium_terms)
            else "LOW"
        )
        # A market-context headline alone must not produce HIGH market risk.
        if is_market and not is_company and severity == "HIGH":
            severity = "MEDIUM"
        if is_company:
            company_matches.append((severity, title[:220]))
        if is_market:
            # The market-context dimension is capped independently, even if a
            # headline also names the company and triggers HIGH company risk.
            market_severity = "MEDIUM" if severity == "HIGH" else severity
            market_matches.append((market_severity, title[:220]))

    rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    def highest(matches: list[tuple[str, str]], default: str = "LOW") -> str:
        return max((entry[0] for entry in matches), key=lambda value: rank[value], default=default)

    company_status = "MATCHES_FOUND_IN_SUPPLIED_FEEDS" if company_matches else "NO_MATCH_IN_SUPPLIED_FEEDS"
    market_status = "MATCHES_FOUND" if market_matches else "NO_MATCH"
    company_risk = highest(company_matches)
    market_risk = highest(market_matches)
    reasons: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for severity, title in company_matches:
        item = (severity, "company-specific", title)
        if item not in seen:
            seen.add(item)
            reasons.append(f"{severity} company-specific: {title}")
    for severity, title in market_matches:
        item = (severity, "market context", title)
        if item not in seen:
            seen.add(item)
            reasons.append(f"{severity} market context: {title}")

    status = "CHECKED_MATCHES" if reasons else "CHECKED_NO_MATCH"
    aggregate = highest([(market_risk, ""), (company_risk, "")])
    return status, aggregate, market_status, market_risk, company_status, company_risk, reasons


def _news_for_symbol(symbol: str, news_items: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """Compatibility wrapper for callers using the original three-value API."""
    status, aggregate, _ms, _mr, _cs, _cr, reasons = _news_risk_breakdown(symbol, news_items)
    return status, aggregate, reasons


def evaluate_symbol_bundle(
    symbol: str,
    bundle: dict[str, Any],
    *,
    index_15m_candles: Iterable[dict[str, Any]] | pd.DataFrame | None = None,
    news_items: list[dict[str, Any]] | None = None,
    margin_per_share: float | None = None,
    now: datetime | None = None,
    settings: RiskSettings = RiskSettings(),
    charge_rates: ChargeRates = ChargeRates(),
    max_candle_age_minutes: int = 15,
) -> ScanCandidate:
    """Evaluate one stock. Returns a candidate/watch/no-trade record, never an order."""
    current = now or datetime.now(IST)
    current = current.astimezone(IST) if current.tzinfo else current.replace(tzinfo=IST)
    symbol = symbol.strip().upper()
    reasons: list[str] = []
    provider = str(bundle.get("provider", "unknown"))
    data_verified_live = bool(bundle.get("verified_live_data", False))
    data_warning = str(bundle.get("data_quality_warning", ""))
    (
        news_status, news_risk, market_context_status, market_context_risk,
        company_headline_match_status, company_specific_risk, news_reasons,
    ) = _news_risk_breakdown(symbol, news_items or [])
    reasons.extend(news_reasons)
    if data_warning:
        reasons.append(data_warning)

    try:
        raw5 = candles_to_frame(bundle.get("candles_5m", []))
        raw15 = candles_to_frame(bundle.get("candles_15m", []))
        c5 = completed_candles(raw5, 5, current)
        c15 = completed_candles(raw15, 15, current)
        idx = completed_candles(candles_to_frame(index_15m_candles), 15, current) if index_15m_candles is not None else pd.DataFrame(columns=OHLCV)
    except (ValueError, TypeError) as exc:
        reasons.append(f"Invalid candle data: {exc}")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, "UNKNOWN", None, None, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)

    if not _is_market_session(current):
        reasons.append("Outside normal NSE cash-market hours or weekend; no current intraday recommendation.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, "UNKNOWN", None, None, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    if c5.empty or c15.empty:
        reasons.append("No completed 5-minute and 15-minute candles are available.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, "UNKNOWN", None, None, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    age = _latest_candle_age_minutes(c5, current, 5)
    if age is None or age > max_candle_age_minutes:
        reasons.append(f"Latest completed 5-minute candle is stale ({age if age is not None else 'unknown'} minutes old).")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, "UNKNOWN", str(c5.index[-1]), None, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)

    ind5 = add_scanner_indicators(c5)
    ind15 = add_scanner_indicators(c15)
    index_indicators = add_scanner_indicators(idx) if not idx.empty else pd.DataFrame()
    regime = classify_market_regime(index_indicators) if not index_indicators.empty else "UNKNOWN"
    stock_regime = _stock_15m_regime(ind15)
    reasons.append(f"15-minute stock trend filter: {stock_regime}.")
    if regime == "UNKNOWN":
        reasons.append("NIFTY 50 regime is unavailable; market-direction confirmation is missing.")
    latest = ind5.iloc[-1]
    atr = float(latest.get("atr_14", float("nan")))
    close = float(latest["close"])
    if not isfinite(atr) or atr <= 0:
        reasons.append("ATR warm-up is incomplete or invalid; no levels generated.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)

    strategy_results = [
        opening_range_signal(symbol, ind5, market_regime=regime),
        vwap_trend_signal(symbol, ind5, market_regime=regime),
        vwap_pullback_signal(symbol, ind5, market_regime=regime),
    ]
    signals = [signal for signal in strategy_results if signal is not None]
    if not signals:
        reasons.append("No strategy produced a valid setup on the latest completed candle.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    sides = {s.side for s in signals}
    if stock_regime == "UNKNOWN":
        reasons.append("Not enough completed 15-minute candles for higher-timeframe confirmation.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    if stock_regime == "CHOPPY":
        reasons.append("15-minute stock trend is mixed; setup suppressed to avoid forcing a trade.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    if len(sides) > 1:
        reasons.append("Strategies disagree on direction; candidate suppressed rather than forcing a trade.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    signal = sorted(signals, key=lambda s: s.confidence, reverse=True)[0]
    if (signal.side == "LONG" and stock_regime != "BULLISH") or (signal.side == "SHORT" and stock_regime != "BEARISH"):
        reasons.append(f"5-minute {signal.side} setup conflicts with the 15-minute stock trend ({stock_regime}); no trade.")
        return ScanCandidate(symbol, "NO_TRADE", None, None, 0, regime, str(ind5.index[-1]), close, None, None, None, 0, None, None, None, False, data_verified_live, news_status, news_risk, reasons,
            market_context_status=market_context_status,
            market_context_risk=market_context_risk,
            company_headline_match_status=company_headline_match_status,
            company_specific_risk=company_specific_risk)
    reasons.append(signal.reason)
    if len(signals) > 1:
        reasons.append(f"{len(signals)} strategies agreed on {signal.side}; strongest-confidence strategy selected.")

    # Wider target spacing is a research starting point because transaction costs
    # are material at this capital size. It must still pass the net-R:R filter.
    levels = plan_atr_levels(signal.side, entry=close, atr=atr, stop_atr_multiple=1.0, target_1_atr_multiple=2.5, target_2_atr_multiple=4.0)
    assessment: TradeAssessment = assess_trade(
        side=signal.side,
        entry_price=levels.entry,
        stop_loss=levels.stop_loss,
        target_1=levels.target_1,
        target_2=levels.target_2,
        settings=settings,
        estimated_margin_per_share=margin_per_share,
        charge_rates=charge_rates,
    )
    margin_verified = margin_per_share is not None and margin_per_share > 0
    if not assessment.accepted:
        reasons.append(assessment.reason)
    if not data_verified_live:
        reasons.append("Data provider has not explicitly verified this as live exchange data; candidate cannot be trade-approved.")
    if news_status in ("NOT_CHECKED", "UNAVAILABLE"):
        reasons.append("News/event checks are incomplete; candidate cannot be trade-approved.")
    if company_specific_risk == "HIGH":
        reasons.append("Potential high-impact company-specific headline; verify the original source before any entry.")
    if company_specific_risk == "MEDIUM":
        reasons.append("Potential company-specific event headline; verify the original source and timing.")
    if market_context_risk == "MEDIUM":
        reasons.append("Potential market-wide event context found; assess relevance to current conditions.")
    if company_headline_match_status == "NO_MATCH_IN_SUPPLIED_FEEDS":
        reasons.append("No company-specific headline matched in supplied RSS items; this does not prove company news is absent.")

    approved = (
        assessment.accepted
        and data_verified_live
        and news_status not in ("NOT_CHECKED", "UNAVAILABLE")
        and company_specific_risk != "HIGH"
    )
    if approved:
        action = "CONDITIONAL_BUY" if signal.side == "LONG" else "CONDITIONAL_SHORT"
        reasons.append("Conditional advisory only: re-check current quote, spread, margin and entry trigger before acting.")
    else:
        action = "WATCH_CANDIDATE"
        reasons.append("Not approved for entry. Wait for all data, news, margin and risk gates to pass.")

    return ScanCandidate(
        symbol=symbol,
        action=action,
        side=signal.side,
        strategy=signal.strategy,
        confidence=signal.confidence,
        market_regime=str(regime),
        signal_time=str(ind5.index[-1]),
        reference_entry=round(levels.entry, 2),
        stop_loss=round(levels.stop_loss, 2),
        target_1=round(levels.target_1, 2),
        target_2=round(levels.target_2, 2),
        quantity=assessment.quantity if approved else 0,
        estimated_net_risk_inr=round(assessment.net_risk_inr, 2) if assessment.accepted else None,
        estimated_net_reward_t1_inr=round(assessment.net_reward_t1_inr, 2) if assessment.accepted else None,
        net_rr_t1=round(assessment.net_rr_t1, 2) if assessment.accepted else None,
        margin_verified=margin_verified,
        data_verified_live=data_verified_live,
        news_status=news_status,
        news_risk=news_risk,
        reasons=reasons,
        market_context_status=market_context_status,
        market_context_risk=market_context_risk,
        company_headline_match_status=company_headline_match_status,
        company_specific_risk=company_specific_risk,
    )


def rank_candidates(candidates: Iterable[ScanCandidate]) -> list[ScanCandidate]:
    """Rank potential setups; approved candidates first, then confidence and risk checks."""
    priority = {"CONDITIONAL_BUY": 0, "CONDITIONAL_SHORT": 0, "WATCH_CANDIDATE": 1, "NO_TRADE": 2}
    return sorted(candidates, key=lambda c: (priority.get(c.action, 3), -c.confidence, c.symbol))
