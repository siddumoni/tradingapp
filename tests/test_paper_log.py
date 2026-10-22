"""Tests for the forward paper-trading log."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from src.orb_backtest import SimConfig, prepare_data, run_backtest
from src.paper_log import (
    FORWARD_MIN_SESSIONS, load_results, plan_session, record_plans, settle_session, summarize_log,
)
from src.strategies.stocks_in_play_orb import SipOrbParams, select_day_plans
from src.synthetic_data import make_synthetic_market

CFG = SimConfig()
PARAMS = SipOrbParams()


@pytest.fixture(scope="module")
def market() -> dict[str, pd.DataFrame]:
    return make_synthetic_market(n_symbols=14, n_sessions=60, seed=21)


def busy_session(market: dict[str, pd.DataFrame]) -> date:
    prepared = prepare_data(market, PARAMS)
    for session in reversed(prepared.sessions()):
        if len(select_day_plans(prepared.features, session, PARAMS)) >= 2:
            return session
    raise AssertionError("no session with plans in fixture")


def opening_only(market: dict[str, pd.DataFrame], session: date) -> dict[str, pd.DataFrame]:
    """What a live run sees at 09:20: full history, but today only the first 5-minute bar."""
    out = {}
    for sym, frame in market.items():
        idx = pd.DatetimeIndex(frame.index)
        today = idx.date == session
        keep = (idx.date < session) | (today & (idx.hour * 60 + idx.minute <= 9 * 60 + 15))
        out[sym] = frame[keep]
    return out


def test_live_plan_from_opening_bar_equals_backtest_plan(market) -> None:
    session = busy_session(market)
    live = plan_session(opening_only(market, session), session, PARAMS)
    prepared = prepare_data(market, PARAMS)
    backtest = select_day_plans(prepared.features, session, PARAMS)
    assert live == backtest and len(live) >= 2


def test_record_is_idempotent_and_stores_params(tmp_path, market) -> None:
    session = busy_session(market)
    log = tmp_path / "log.jsonl"
    plans = plan_session(opening_only(market, session), session, PARAMS)
    assert record_plans(log, plans, PARAMS) == len(plans)
    assert record_plans(log, plans, PARAMS) == 0
    first = json.loads(log.read_text().splitlines()[0])
    assert first["type"] == "plan" and first["params"]["stop_atr_fraction"] == PARAMS.stop_atr_fraction


def test_settlement_reproduces_the_backtest_exactly(tmp_path, market) -> None:
    session = busy_session(market)
    log = tmp_path / "log.jsonl"
    record_plans(log, plan_session(opening_only(market, session), session, PARAMS), PARAMS)
    written = settle_session(log, market, session, CFG)
    assert written and all(w["outcome"] for w in written)
    expected = run_backtest(prepare_data(market, PARAMS), PARAMS, CFG, [session]).trades
    got = load_results(log)
    assert len(got) == len(expected)
    for col in ("symbol", "side", "entry", "exit", "quantity", "net_pnl", "exit_reason"):
        assert list(got.sort_values("symbol")[col]) == list(expected.sort_values("symbol")[col])
    # idempotent
    assert settle_session(log, market, session, CFG) == []


def test_settlement_uses_logged_params_not_current_defaults(tmp_path, market) -> None:
    session = busy_session(market)
    log = tmp_path / "log.jsonl"
    wide = SipOrbParams(stop_atr_fraction=0.5)
    record_plans(log, plan_session(opening_only(market, session), session, wide), wide)
    settle_session(log, market, session, CFG)
    trades = load_results(log)
    expected = run_backtest(prepare_data(market, wide), wide, CFG, [session]).trades
    assert list(trades.sort_values("symbol")["net_pnl"]) == list(expected.sort_values("symbol")["net_pnl"])


def test_incomplete_session_is_left_unsettled_for_retry(tmp_path, market) -> None:
    session = busy_session(market)
    log = tmp_path / "log.jsonl"
    record_plans(log, plan_session(opening_only(market, session), session, PARAMS), PARAMS)
    assert settle_session(log, opening_only(market, session), session, CFG) == []   # data stops at 09:15
    assert settle_session(log, market, session, CFG) != []                           # retry with full data works


def test_settle_with_no_plans_is_a_noop(tmp_path, market) -> None:
    assert settle_session(tmp_path / "none.jsonl", market, date(2024, 1, 1), CFG) == []


def test_summary_collecting_then_reports_metrics(tmp_path, market) -> None:
    log = tmp_path / "log.jsonl"
    assert summarize_log(log, 20_000.0)["status"] == "COLLECTING"
    prepared = prepare_data(market, PARAMS)
    with_plans = 0
    for session in prepared.sessions()[:6]:
        plans = plan_session(opening_only(market, session), session, PARAMS)
        with_plans += bool(plans)          # most days have no stocks "in play" and so no plans
        record_plans(log, plans, PARAMS)
        settle_session(log, market, session, CFG)
    summary = summarize_log(log, 20_000.0, n_boot=100)
    assert 0 < with_plans < 6               # fixture really contains quiet days
    assert summary["sessions_settled"] == with_plans < FORWARD_MIN_SESSIONS
    assert summary["status"] == "COLLECTING"
    assert summary["metrics"]["n_trades"] > 0 and summary["advisory_only"] is True
    json.dumps(summary)  # JSON-safe
