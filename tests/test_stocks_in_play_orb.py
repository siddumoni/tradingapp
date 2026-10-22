"""Tests for Stocks-in-Play ORB features and plan selection."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.strategies.stocks_in_play_orb import SipOrbParams, compute_features, select_day_plans

IST = "Asia/Kolkata"


def make_frame(days: list[list[tuple]], start: str = "2024-01-01") -> pd.DataFrame:
    """days: list of sessions, each a list of (open, high, low, close, volume) from 09:15."""
    rows, index = [], []
    for day, bars in zip(pd.bdate_range(start=start, periods=len(days)), days):
        for i, bar in enumerate(bars):
            index.append(pd.Timestamp(day).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15 + 5 * i))
            rows.append(bar)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(index))


def flat_day(volume: float = 100.0, price: float = 100.0, bars: int = 6) -> list[tuple]:
    # every bar: o=c=price, range 2 -> daily range 2, true range 2 when there is no gap
    return [(price, price + 1, price - 1, price, volume)] * bars


PARAMS = SipOrbParams(rvol_lookback=14, atr_lookback=14)


def test_params_validate() -> None:
    with pytest.raises(ValueError):
        SipOrbParams(opening_bars=0)
    with pytest.raises(ValueError):
        SipOrbParams(stop_atr_fraction=0)
    with pytest.raises(ValueError):
        SipOrbParams(top_n=0)
    with pytest.raises(ValueError):
        SipOrbParams(max_cost_fraction_of_risk=1.5)


def test_rvol_and_atr_values() -> None:
    days = [flat_day(volume=100) for _ in range(15)] + [flat_day(volume=300)]
    feats = compute_features(make_frame(days), PARAMS)
    today = feats.iloc[-1]
    assert today["rvol"] == pytest.approx(3.0)
    assert today["atr"] == pytest.approx(2.0)
    # Not enough history for the first 14 sessions
    assert feats["rvol"].iloc[:14].isna().all()


def test_features_have_no_lookahead() -> None:
    base = [flat_day(volume=100) for _ in range(20)]
    feats_a = compute_features(make_frame(base), PARAMS)

    # 1) Changing a LATER session must not change an earlier session's features.
    later = [list(d) for d in base]
    later[19] = flat_day(volume=9999, price=500)
    feats_b = compute_features(make_frame(later), PARAMS)
    earlier = feats_a.index[17]
    assert feats_a.loc[earlier, ["rvol", "atr"]].equals(feats_b.loc[earlier, ["rvol", "atr"]])

    # 2) Changing today's bars AFTER the opening bar must not change today's rvol or ATR.
    changed = [list(d) for d in base]
    changed[19] = [changed[19][0]] + [(100.0, 400.0, 1.0, 250.0, 1.0)] * 5
    feats_c = compute_features(make_frame(changed), PARAMS)
    last = feats_a.index[19]
    assert feats_a.loc[last, "rvol"] == pytest.approx(feats_c.loc[last, "rvol"])
    assert feats_a.loc[last, "atr"] == pytest.approx(feats_c.loc[last, "atr"])


def test_session_without_expected_opening_bar_is_dropped() -> None:
    frame = make_frame([flat_day() for _ in range(3)])
    # remove the 09:15 bar of the second session
    second = frame.index[pd.DatetimeIndex(frame.index).date == frame.index[6].date()][0]
    frame = frame.drop(second)
    feats = compute_features(frame, SipOrbParams(rvol_lookback=2, atr_lookback=2))
    assert len(feats) == 2


def test_multi_bar_opening_range_requires_consecutive_bars() -> None:
    params = SipOrbParams(opening_bars=3, rvol_lookback=2, atr_lookback=2)
    frame = make_frame([flat_day() for _ in range(3)])
    assert len(compute_features(frame, params)) == 3
    gap = frame.drop(frame.index[7])  # drop 09:20 of the second session
    assert len(compute_features(gap, params)) == 2


def _feats(symbol_rows: dict[str, dict]) -> dict[str, pd.DataFrame]:
    session = date(2024, 2, 1)
    out = {}
    for sym, row in symbol_rows.items():
        base = {"or_open": 100.0, "or_close": 101.0, "or_high": 102.0, "or_low": 99.0,
                "or_volume": 1.0, "day_high": 0.0, "day_low": 0.0, "day_close": 0.0,
                "rvol": 2.0, "atr": 3.0}
        base.update(row)
        out[sym] = pd.DataFrame([base], index=pd.Index([session], name="session"))
    return out


def test_selection_ranks_by_rvol_and_applies_top_n() -> None:
    feats = _feats({"AAA": {"rvol": 2.0}, "BBB": {"rvol": 5.0}, "CCC": {"rvol": 3.0}, "DDD": {"rvol": 1.0}})
    plans = select_day_plans(feats, date(2024, 2, 1), SipOrbParams(top_n=2, min_rvol=1.5))
    assert [p.symbol for p in plans] == ["BBB", "CCC"]  # DDD below min_rvol; AAA cut by top_n


def test_selection_direction_trigger_and_stop() -> None:
    feats = _feats({
        "UP": {"or_open": 100.0, "or_close": 101.0},
        "DOWN": {"or_open": 100.0, "or_close": 99.0, "rvol": 2.5},
        "DOJI": {"or_open": 100.0, "or_close": 100.0, "rvol": 9.0},
    })
    plans = {p.symbol: p for p in select_day_plans(feats, date(2024, 2, 1), SipOrbParams(stop_atr_fraction=0.25))}
    assert "DOJI" not in plans
    assert plans["UP"].side == "LONG" and plans["UP"].trigger == 102.0
    assert plans["DOWN"].side == "SHORT" and plans["DOWN"].trigger == 99.0
    assert plans["UP"].stop_distance == pytest.approx(0.75)


def test_selection_filters_price_atr_and_missing_values() -> None:
    feats = _feats({
        "CHEAP": {"or_close": 20.0, "or_open": 19.0},          # below min_price
        "QUIET": {"atr": 0.5, "or_close": 101.0},              # ATR 0.5% of price < 1%
        "NOHIST": {"rvol": float("nan")},
        "GOOD": {},
    })
    plans = select_day_plans(feats, date(2024, 2, 1), SipOrbParams())
    assert [p.symbol for p in plans] == ["GOOD"]


def test_side_override_for_placebo_changes_only_direction() -> None:
    feats = _feats({"AAA": {}})
    p = select_day_plans(feats, date(2024, 2, 1), SipOrbParams(), side_override={"AAA": "SHORT"})[0]
    assert p.side == "SHORT" and p.trigger == 99.0
