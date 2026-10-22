"""Tests for the Stocks-in-Play ORB simulator using hand-built bars."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.orb_backtest import DayBars, SimConfig, prepare_data, run_backtest, simulate_plan
from src.strategies.stocks_in_play_orb import DayPlan, SipOrbParams
from src.synthetic_data import make_synthetic_market

IST = "Asia/Kolkata"
# Short session so hand-made bars stay readable: entries allowed before 09:40,
# squared off at the close of the 09:50 bar (bars 09:15..09:50 = 8 bars).
PARAMS = SipOrbParams(entry_cutoff="09:40", square_off_bar="09:50", top_n=5)
CFG = SimConfig()


def day(bars: list[tuple[float, float, float, float]]) -> DayBars:
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-02-01").tz_localize(IST) + pd.Timedelta(hours=9, minutes=15 + 5 * i)
        for i in range(len(bars))
    ])
    arr = np.array(bars, dtype=float)
    return DayBars(index=idx, minutes=(idx.hour * 60 + idx.minute).to_numpy(),
                   open=arr[:, 0], high=arr[:, 1], low=arr[:, 2], close=arr[:, 3])


def plan(side: str = "LONG", trigger: float = 100.0, stop_distance: float = 2.0) -> DayPlan:
    return DayPlan(symbol="TEST", session=date(2024, 2, 1), side=side, trigger=trigger,
                   stop_distance=stop_distance, rvol=3.0, atr=8.0, or_high=trigger, or_low=trigger - 1)


QUIET = (101.0, 102.0, 100.5, 101.5)  # stays well clear of stop 98 and trigger 100 for a LONG


def test_long_normal_trade_exits_at_square_off() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 100.5, 99.5, 100.2)] + [QUIET] * 5 + [(102, 103.5, 101.8, 103.0)]
    trade, reason = simulate_plan(plan(), day(bars), PARAMS, CFG)
    assert reason == "traded"
    assert trade["entry"] == 100.0 and trade["exit"] == 103.0
    assert trade["exit_reason"] == "square_off"
    assert trade["quantity"] == 40  # min(risk 200/2=100, cap 20000/5/100=40)
    assert trade["gross_pnl"] == pytest.approx(120.0)
    assert trade["slippage"] == pytest.approx((100 + 103) * 40 * 5 / 10_000)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["costs"] - trade["slippage"])
    assert trade["r_multiple"] == pytest.approx(trade["net_pnl"] / (2.0 * 40))


def test_gap_through_trigger_fills_at_open() -> None:
    bars = [(99, 100, 98.5, 99.8), (101.0, 101.5, 100.8, 101.2)] + [QUIET] * 6
    trade, _ = simulate_plan(plan(), day(bars), PARAMS, CFG)
    assert trade["entry"] == 101.0          # not the 100.0 trigger
    assert trade["stop"] == pytest.approx(99.0)


def test_stop_hit_exits_at_stop_with_loss_beyond_gross_risk() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 100.5, 99.5, 100.2), (100.2, 100.4, 97.5, 98.0)] + [QUIET] * 5
    trade, _ = simulate_plan(plan(), day(bars), PARAMS, CFG)
    assert trade["exit_reason"] == "stop" and trade["exit"] == pytest.approx(98.0)
    assert trade["gross_pnl"] == pytest.approx(-2.0 * 40)
    assert trade["net_pnl"] < trade["gross_pnl"]        # costs and slippage only ever hurt
    assert trade["r_multiple"] < -1.0                    # realised loss exceeds 1R after costs


def test_entry_bar_that_also_reaches_stop_is_assumed_stopped() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 101.0, 97.0, 100.5)] + [QUIET] * 6
    trade, _ = simulate_plan(plan(), day(bars), PARAMS, CFG)
    assert trade["exit_reason"] == "stop" and trade["entry_time"] == trade["exit_time"]


def test_later_gap_below_stop_exits_at_open_not_stop() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 100.5, 99.5, 100.2), (96.0, 97.0, 95.0, 96.5)] + [QUIET] * 5
    trade, _ = simulate_plan(plan(), day(bars), PARAMS, CFG)
    assert trade["exit_reason"] == "stop_gap" and trade["exit"] == 96.0


def test_short_mirrors_long() -> None:
    bars = [(101, 101.5, 100, 100.2), (100.1, 100.4, 99.5, 99.8)] + [(99.0, 99.5, 98.5, 99.0)] * 5 + [(98.0, 98.2, 96.5, 97.0)]
    trade, _ = simulate_plan(plan("SHORT", trigger=100.0), day(bars), PARAMS, CFG)
    assert trade["side"] == "SHORT" and trade["entry"] == 100.0 and trade["exit"] == 97.0
    assert trade["gross_pnl"] == pytest.approx(3.0 * 40)
    stopped = [(101, 101.5, 100, 100.2), (100.1, 100.4, 99.5, 99.8), (100.0, 102.5, 99.9, 102.2)] + [QUIET] * 5
    trade, _ = simulate_plan(plan("SHORT", trigger=100.0), day(stopped), PARAMS, CFG)
    assert trade["exit_reason"] == "stop" and trade["exit"] == pytest.approx(102.0)


def test_no_trigger_and_entry_after_cutoff_are_not_taken() -> None:
    never = [(99, 100, 98.5, 99.8)] + [(99.0, 99.8, 98.5, 99.0)] * 7
    assert simulate_plan(plan(), day(never), PARAMS, CFG) == (None, "no_trigger")
    late = [(99, 100, 98.5, 99.8)] + [(99.0, 99.8, 98.5, 99.0)] * 5 + [(99.5, 101.0, 99.0, 100.8)] * 2
    assert simulate_plan(plan(), day(late), PARAMS, CFG)[1] == "no_trigger"   # first touch at/after 09:40


def test_sizing_cap_and_screens() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 100.5, 99.5, 100.2)] + [QUIET] * 6
    # price too high for the per-position cap -> zero shares
    assert simulate_plan(plan(trigger=5000.0), day(bars), PARAMS, CFG) == (None, "qty_below_1")
    # tiny stop: costs dwarf the risk -> cost screen rejects
    assert simulate_plan(plan(stop_distance=0.2), day(bars), PARAMS, CFG) == (None, "cost_filter")
    # leverage raises the cap but never the risk-based quantity
    levered = SimConfig(leverage=5.0)
    trade, _ = simulate_plan(plan(stop_distance=4.0), day(bars), PARAMS, levered)
    assert trade["quantity"] == 50  # risk 200/4 = 50 < cap 100


def test_higher_slippage_never_improves_a_trade() -> None:
    bars = [(99, 100, 98.5, 99.8), (99.9, 100.5, 99.5, 100.2)] + [QUIET] * 5 + [(102, 103.5, 101.8, 103.0)]
    low, _ = simulate_plan(plan(), day(bars), PARAMS, SimConfig(slippage_bps_per_side=1.0))
    high, _ = simulate_plan(plan(), day(bars), PARAMS, SimConfig(slippage_bps_per_side=10.0))
    assert high["net_pnl"] < low["net_pnl"]


def test_run_backtest_is_deterministic_and_placebo_changes_only_direction() -> None:
    frames = make_synthetic_market(n_symbols=12, n_sessions=60, seed=3)
    params = SipOrbParams()
    prepared = prepare_data(frames, params)
    a = run_backtest(prepared, params, CFG)
    b = run_backtest(prepared, params, CFG)
    assert len(a.trades) > 20
    assert a.trades.equals(b.trades) and a.plans_made == b.plans_made
    p1 = run_backtest(prepared, params, CFG, rng=np.random.default_rng(5))
    p2 = run_backtest(prepared, params, CFG, rng=np.random.default_rng(5))
    assert p1.trades.equals(p2.trades)                       # seeded placebo is reproducible
    assert p1.plans_made == a.plans_made                     # same symbols/days are considered
    assert set(p1.trades["side"]) == {"LONG", "SHORT"}       # directions are randomised...
    assert not p1.trades[["symbol", "session", "side"]].equals(a.trades[["symbol", "session", "side"]])


def test_session_subset_only_trades_requested_sessions() -> None:
    frames = make_synthetic_market(n_symbols=12, n_sessions=60, seed=3)
    params = SipOrbParams()
    prepared = prepare_data(frames, params)
    subset = prepared.sessions()[-15:]
    result = run_backtest(prepared, params, CFG, subset)
    assert len(result.trades) > 0
    assert set(result.trades["session"]) <= set(subset)
    assert result.sessions == subset
