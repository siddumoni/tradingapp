from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.market_scanner import add_scanner_indicators, candles_to_frame, completed_candles, evaluate_symbol_bundle, _news_for_symbol, _news_risk_breakdown

IST = ZoneInfo("Asia/Kolkata")


def sample_candles(count=40, start="2026-10-16 09:15:00"):
    idx = pd.date_range(start=start, periods=count, freq="5min", tz=IST)
    rows = []
    for i, stamp in enumerate(idx):
        close = 100 + i * 0.2
        rows.append({"date": stamp, "open": close - 0.1, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000 + i * 10})
    return rows


def test_candles_to_frame_normalizes_timezone_and_columns():
    frame = candles_to_frame(sample_candles(3))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index.tz) == "Asia/Kolkata"


def test_indicators_are_computed_without_external_ta_package():
    frame = add_scanner_indicators(candles_to_frame(sample_candles(40)))
    for col in ("ema_9", "ema_21", "rsi_14", "macd_histogram", "atr_14", "vwap", "volume_sma_20"):
        assert col in frame.columns
    assert frame["atr_14"].dropna().gt(0).any()


def test_completed_candles_excludes_current_forming_bar():
    now = datetime(2026, 10, 16, 9, 42, tzinfo=IST)
    frame = candles_to_frame(sample_candles(6, start="2026-10-16 09:15:00"))
    done = completed_candles(frame, 5, now)
    assert done.index.max().strftime("%H:%M") == "09:35"


def test_weekend_fails_closed_without_signal():
    now = datetime(2026, 10, 18, 13, 30, tzinfo=IST)  # Sunday
    bundle = {"provider": "test", "candles_5m": sample_candles(), "candles_15m": sample_candles()}
    result = evaluate_symbol_bundle("RELIANCE", bundle, now=now)
    assert result.action == "NO_TRADE"
    assert result.quantity == 0
    assert any("Outside normal NSE cash-market hours" in reason for reason in result.reasons)


def test_unverified_provider_never_approves_trade():
    now = datetime(2026, 10, 16, 11, 0, tzinfo=IST)
    c5 = sample_candles(60, start="2026-10-16 09:15:00")
    # Rebuild a plausible 15-minute series with timestamps at 15-minute intervals.
    c15 = sample_candles(24, start="2026-10-16 09:15:00")
    bundle = {"provider": "unofficial-test", "candles_5m": c5, "candles_15m": c15, "data_quality_warning": "unofficial"}
    news = [{"title": "Indian stock market opens steady", "summary": "NIFTY in focus"}]
    result = evaluate_symbol_bundle("RELIANCE", bundle, index_15m_candles=c15, news_items=news, margin_per_share=100, now=now)
    assert result.action != "CONDITIONAL_BUY"
    assert result.action != "CONDITIONAL_SHORT"
    assert result.quantity == 0


def test_news_triage_ignores_encoded_html_and_url_keywords():
    items = [{
        "title": "Reliance shares trade steady",
        "summary": '<a href="https://example.com/raid-investigation-penalty">Read more</a>',
    }]
    status, risk, reasons = _news_for_symbol("RELIANCE", items)
    assert status == "CHECKED_MATCHES"
    assert risk == "LOW"
    assert reasons and "Reliance shares trade steady" in reasons[0]


def test_news_triage_detects_high_risk_terms_in_readable_headline():
    items = [{"title": "Reliance faces regulatory action", "summary": "Company update"}]
    status, risk, _ = _news_for_symbol("RELIANCE", items)
    assert status == "CHECKED_MATCHES"
    assert risk == "HIGH"


def test_market_wide_high_keyword_is_capped_at_medium_for_each_stock():
    items = [{
        "title": "Sensex slips as investigation widens across markets",
        "summary": '<a href="https://example.com/fraud-raid-penalty">Read more</a>',
    }]
    status, risk, reasons = _news_for_symbol("RELIANCE", items)
    assert status == "CHECKED_MATCHES"
    assert risk == "MEDIUM"
    assert reasons and "market context" in reasons[0]


def test_company_specific_high_risk_headline_remains_high_with_company_alias():
    items = [{"title": "HDFC Bank faces regulatory action", "summary": "Company update"}]
    status, risk, reasons = _news_for_symbol("HDFCBANK", items)
    assert status == "CHECKED_MATCHES"
    assert risk == "HIGH"
    assert reasons and "company" in reasons[0]


def test_unrelated_company_headline_does_not_match_requested_stock():
    items = [{"title": "Tata Motors faces fraud probe", "summary": "Company update"}]
    status, risk, reasons = _news_for_symbol("RELIANCE", items)
    assert status == "CHECKED_NO_MATCH"
    assert risk == "LOW"
    assert reasons == []


def test_company_specific_alias_can_match_non_high_headline():
    items = [{"title": "Infosys announces quarterly results", "summary": "Company update"}]
    status, risk, _ = _news_for_symbol("INFY", items)
    assert status == "CHECKED_MATCHES"
    assert risk == "MEDIUM"



def test_market_context_and_company_specific_risk_are_separate():
    items = [
        {"title": "Rupee rises as crude oil pressure continues", "summary": "Indian market context"},
        {"title": "Reliance shares trade steady", "summary": "Company update"},
    ]
    status, aggregate, market_status, market_risk, company_status, company_risk, reasons = _news_risk_breakdown("RELIANCE", items)
    assert status == "CHECKED_MATCHES"
    assert market_status == "MATCHES_FOUND"
    assert market_risk == "MEDIUM"
    assert company_status == "MATCHES_FOUND_IN_SUPPLIED_FEEDS"
    assert company_risk == "LOW"
    assert any("market context" in reason for reason in reasons)
    assert any("company-specific" in reason for reason in reasons)


def test_market_wide_high_keyword_does_not_create_company_specific_high_risk():
    items = [{
        "title": "Sensex slips as investigation widens across markets",
        "summary": "Indian stock market context",
    }]
    _, _, _, market_risk, company_status, company_risk, _ = _news_risk_breakdown("RELIANCE", items)
    assert market_risk == "MEDIUM"
    assert company_status == "NO_MATCH_IN_SUPPLIED_FEEDS"
    assert company_risk == "LOW"


def test_company_specific_high_risk_is_separate_from_market_context():
    items = [
        {"title": "Reliance faces regulatory action", "summary": "Company update"},
        {"title": "NIFTY tracks crude oil volatility", "summary": "Market context"},
    ]
    _, _, _, market_risk, company_status, company_risk, _ = _news_risk_breakdown("RELIANCE", items)
    assert market_risk == "MEDIUM"
    assert company_status == "MATCHES_FOUND_IN_SUPPLIED_FEEDS"
    assert company_risk == "HIGH"


def test_no_headlines_means_news_coverage_unavailable_not_no_match():
    result = _news_risk_breakdown("RELIANCE", [])
    assert result[0] == "UNAVAILABLE"
    assert result[2] == "UNAVAILABLE"
    assert result[4] == "UNAVAILABLE"



def test_high_company_headline_does_not_raise_market_context_dimension_to_high():
    items = [{
        "title": "Reliance faces regulatory action",
        "summary": "NIFTY and Indian stock market reaction",
    }]
    _, _, _, market_risk, _, company_risk, _ = _news_risk_breakdown("RELIANCE", items)
    assert company_risk == "HIGH"
    assert market_risk == "MEDIUM"
