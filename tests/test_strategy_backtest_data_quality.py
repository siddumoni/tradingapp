"""Tests for incomplete-session handling in the intraday backtest pipeline."""
import pandas as pd
import pytest

from src.strategy_backtest import filter_incomplete_sessions


def make_session(day: str) -> pd.DataFrame:
    index = pd.date_range(
        f"{day} 09:15", f"{day} 15:10", freq="5min",
        tz="Asia/Kolkata",
    )
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        index=index,
    )


def test_keeps_complete_session_and_reports_one_excluded_session():
    complete = make_session("2026-10-14")
    incomplete = make_session("2026-10-15").drop(
        pd.Timestamp("2026-10-15 11:45", tz="Asia/Kolkata")
    )
    frame = pd.concat([complete, incomplete])
    filtered, report = filter_incomplete_sessions(frame)
    assert len(filtered) == 72
    assert set(filtered.index.date) == {pd.Timestamp("2026-10-14").date()}
    assert report["sessions_seen"] == 2
    assert report["sessions_kept"] == 1
    assert report["sessions_excluded"] == 1
    assert report["excluded_sessions"][0]["session"] == "2026-10-15"
    assert "11:45" in report["excluded_sessions"][0]["missing_bars"]


def test_rejects_duplicate_timestamps():
    complete = make_session("2026-10-14")
    with pytest.raises(ValueError, match="Duplicate candle timestamps"):
        filter_incomplete_sessions(pd.concat([complete, complete.iloc[[0]]]))


def test_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Missing candle columns"):
        filter_incomplete_sessions(make_session("2026-10-14").drop(columns=["volume"]))


def test_empty_input_returns_zeroed_diagnostics():
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    filtered, report = filter_incomplete_sessions(frame)
    assert filtered.empty
    assert report["sessions_seen"] == 0
    assert report["sessions_excluded"] == 0
