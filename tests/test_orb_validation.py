"""Tests for the ORB validation gate.

End-to-end tests use synthetic markets whose truth is known. The planted-edge test
checks the harness can *detect* an edge that is really there; the null tests check
it does not invent one. Neither says anything about real markets.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.orb_validation import (
    GateCriteria, bootstrap_mean_ci, compute_metrics, evaluate_gate, run_validation, split_sessions,
)
from src.synthetic_data import make_synthetic_market

SMALL = GateCriteria(min_holdout_trades=60, min_holdout_sessions=25, min_dev_trades=20)


def trades_frame(pnls: list[float], per_day: int = 1) -> pd.DataFrame:
    rows = []
    for i, pnl in enumerate(pnls):
        rows.append({
            "symbol": f"S{i}", "session": date(2024, 1, 1) + timedelta(days=i // per_day),
            "entry_time": f"2024-01-01 09:{i % 60:02d}", "net_pnl": pnl, "gross_pnl": pnl + 3.0,
            "costs": 2.0, "slippage": 1.0, "r_multiple": pnl / 10.0,
            "exit_reason": "stop" if pnl < 0 else "square_off",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- split
def test_split_is_chronological_disjoint_and_holdout_is_latest() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    dev, hold = split_sessions(list(reversed(sessions)), 0.4)  # input order must not matter
    assert dev + hold == sessions
    assert len(hold) == 40 and not set(dev) & set(hold)
    assert max(dev) < min(hold)


def test_split_validation() -> None:
    with pytest.raises(ValueError):
        split_sessions([date(2024, 1, 1)], 0.0)
    with pytest.raises(ValueError):
        split_sessions([date(2024, 1, 1)], 1.0)
    assert split_sessions([date(2024, 1, 1)], 0.5) == ([date(2024, 1, 1)], [])


# ---------------------------------------------------------------- metrics
def test_metrics_hand_computed() -> None:
    m = compute_metrics(trades_frame([10.0, -5.0, -5.0, 20.0]), capital=100.0)
    assert m["n_trades"] == 4 and m["win_rate"] == 0.5
    assert m["mean_net_pnl"] == pytest.approx(5.0) and m["total_net_pnl"] == pytest.approx(20.0)
    assert m["profit_factor"] == pytest.approx(30.0 / 10.0)
    # cumulative: 10, 5, 0, 20 -> peak 10, trough 0 => drawdown 10
    assert m["max_drawdown_inr"] == pytest.approx(10.0) and m["max_drawdown_fraction"] == pytest.approx(0.10)
    assert m["stop_exit_rate"] == pytest.approx(0.5)
    assert m["total_costs"] == pytest.approx(12.0)


def test_metrics_edge_cases() -> None:
    assert compute_metrics(pd.DataFrame(), 100.0)["n_trades"] == 0
    assert compute_metrics(trades_frame([5.0, 6.0]), 100.0)["profit_factor"] is None  # no losing trades
    assert compute_metrics(trades_frame([-5.0, -6.0]), 100.0)["max_drawdown_inr"] == pytest.approx(11.0)


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    rng_pnl = np.random.default_rng(0)
    trades = trades_frame(list(rng_pnl.normal(5.0, 20.0, 400)), per_day=4)
    a = bootstrap_mean_ci(trades, 1000, np.random.default_rng(1), 0.10)
    b = bootstrap_mean_ci(trades, 1000, np.random.default_rng(1), 0.10)
    assert a == b
    assert a[0] < trades["net_pnl"].mean() < a[1]


def test_bootstrap_detects_clear_edge_and_straddles_zero_for_noise() -> None:
    clear = trades_frame([8.0, 12.0] * 100, per_day=2)
    assert bootstrap_mean_ci(clear, 1000, np.random.default_rng(2), 0.10)[0] > 0
    noise = trades_frame(list(np.random.default_rng(3).normal(0.0, 20.0, 300)), per_day=3)
    low, high = bootstrap_mean_ci(noise, 1000, np.random.default_rng(4), 0.10)
    assert low < 0 < high


def test_bootstrap_with_too_few_sessions_is_nan() -> None:
    low, high = bootstrap_mean_ci(trades_frame([1.0, 2.0, 3.0]), 200, np.random.default_rng(0), 0.10)
    assert np.isnan(low) and np.isnan(high)


def test_bootstrap_resamples_whole_days() -> None:
    # One huge winning day among many small losing days: a day-block bootstrap must give a
    # much wider interval than treating the same trades as independent.
    pnls = [-1.0] * 60 + [200.0] * 5
    clustered = trades_frame(pnls, per_day=5)           # the 5 winners share one day
    spread = trades_frame(pnls, per_day=1)               # every trade on its own day
    lo_c, hi_c = bootstrap_mean_ci(clustered, 2000, np.random.default_rng(5), 0.10)
    lo_s, hi_s = bootstrap_mean_ci(spread, 2000, np.random.default_rng(5), 0.10)
    assert (hi_c - lo_c) > (hi_s - lo_s)


# ---------------------------------------------------------------- gate logic
GOOD_HOLD = {"n_trades": 200, "mean_net_pnl": 10.0, "profit_factor": 1.5, "max_drawdown_fraction": 0.1}
GOOD_DEV = {"mean_net_pnl": 8.0}


def gate(**over):
    hold = {**GOOD_HOLD, **over.pop("hold", {})}
    dev = {**GOOD_DEV, **over.pop("dev", {})}
    args = {"ci_low": 2.0, "p_value": 0.01, "hold_sessions": 60, **over}
    return evaluate_gate(dev, hold, args["ci_low"], args["p_value"], args["hold_sessions"], GateCriteria())


def test_gate_passes_only_when_every_criterion_is_met() -> None:
    assert gate()[0] == "PASS_HISTORICAL_GATE"


@pytest.mark.parametrize("override,expected", [
    ({"hold": {"n_trades": 99}}, "INSUFFICIENT_DATA"),
    ({"hold_sessions": 39}, "INSUFFICIENT_DATA"),
    ({"dev": {"mean_net_pnl": -1.0}}, "FAIL"),
    ({"hold": {"mean_net_pnl": -0.5}}, "FAIL"),
    ({"ci_low": 0.0}, "FAIL"),
    ({"ci_low": float("nan")}, "FAIL"),
    ({"p_value": 0.06}, "FAIL"),
    ({"p_value": None}, "FAIL"),
    ({"hold": {"profit_factor": 1.1}}, "FAIL"),
    ({"hold": {"profit_factor": None}}, "FAIL"),
    ({"hold": {"max_drawdown_fraction": 0.30}}, "FAIL"),
])
def test_each_criterion_can_veto(override, expected) -> None:
    verdict, reasons = gate(**override)
    assert verdict == expected and reasons


def test_gate_reports_all_failures_not_just_the_first() -> None:
    verdict, reasons = gate(ci_low=-1.0, p_value=0.5, hold={"profit_factor": 0.9})
    assert verdict == "FAIL" and len(reasons) >= 3


# ---------------------------------------------------------------- end to end
def test_planted_edge_is_detected() -> None:
    """Edge strength/seed were chosen so the harness has enough power at this small size."""
    frames = make_synthetic_market(n_symbols=14, n_sessions=170, seed=1, edge=0.05)
    report = run_validation(frames, criteria=SMALL, n_boot=500, n_placebo=60, seed=7)
    assert report["verdict"] == "PASS_HISTORICAL_GATE", report["reasons"]
    assert report["holdout"]["mean_net_pnl"] > 0
    assert report["holdout_bootstrap_ci_mean_net_pnl"][0] > 0
    assert report["placebo"]["p_value"] <= 0.05


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pure_noise_never_passes(seed: int) -> None:
    """Seeds fixed in advance (1-3); results are not cherry-picked."""
    frames = make_synthetic_market(n_symbols=14, n_sessions=170, seed=seed, edge=0.0)
    report = run_validation(frames, criteria=SMALL, n_boot=500, n_placebo=30, seed=7)
    assert report["verdict"] in {"FAIL", "INSUFFICIENT_DATA"}


def test_report_is_json_safe_advisory_only_and_flags_limits() -> None:
    frames = make_synthetic_market(n_symbols=10, n_sessions=90, seed=5)
    report = run_validation(frames, criteria=SMALL, n_boot=200, n_placebo=10, seed=1)
    text = json.dumps(report)                      # must not raise (dates, NaN, numpy)
    assert report["advisory_only"] is True and report["orders_placed_or_modified"] is False
    assert "NaN" not in text and "Infinity" not in text
    assert report["sessions_dev"] + report["sessions_holdout"] == report["sessions_total"]


def test_too_little_data_is_insufficient_not_a_verdict() -> None:
    frames = make_synthetic_market(n_symbols=5, n_sessions=18, seed=2)  # barely past the warm-up
    report = run_validation(frames, criteria=SMALL, n_boot=100, n_placebo=5, seed=1)
    assert report["verdict"] == "INSUFFICIENT_DATA"


def test_holdout_data_cannot_influence_development_choices() -> None:
    """Replace the holdout period with a completely different random path; the development
    results and the parameter chosen on them must be bit-for-bit unchanged."""
    a = make_synthetic_market(n_symbols=10, n_sessions=120, seed=11)
    other = make_synthetic_market(n_symbols=10, n_sessions=120, seed=12)
    b = {}
    for sym, frame in a.items():
        cut = frame.index[-40 * 75]                      # last 40 sessions swapped out (holdout is ~42)
        b[sym] = pd.concat([frame[frame.index < cut], other[sym][other[sym].index >= cut]])
    ra = run_validation(a, criteria=SMALL, n_boot=100, n_placebo=5, seed=1, holdout_fraction=0.4)
    rb = run_validation(b, criteria=SMALL, n_boot=100, n_placebo=5, seed=1, holdout_fraction=0.4)
    assert ra["sessions_holdout"] >= 40                  # the swap really lies inside the holdout
    assert ra["sessions_dev"] == rb["sessions_dev"]
    assert ra["dev_grid"] == rb["dev_grid"]
    assert ra.get("chosen") == rb.get("chosen")
    assert ra["holdout"] != rb["holdout"]                # ...and the swap does change the holdout
