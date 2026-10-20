import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, backtest_fixed_levels, summarize_trades


def make_candles():
    idx = pd.date_range("2026-10-12 09:15", periods=8, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({
        "open": [100,100,100,101,102,103,101,102],
        "high": [101,101,102,103,104,104,103,103],
        "low":  [99,99,99,100,101,100,100,101],
        "close":[100,100,101,102,103,101,102,102],
        "volume":[1000]*8,
    }, index=idx)


def one_signal(history, i):
    if i == 1:
        return {"side":"LONG", "stop_loss":98.0, "target":105.0}
    return None


def test_backtest_returns_metrics_and_trades():
    trades, summary = backtest_fixed_levels("TEST", make_candles(), one_signal)
    assert not trades.empty
    assert summary["trades"] == len(trades)
    assert "net_pnl" in trades.columns
    assert "max_drawdown" in summary


def test_backtest_empty_input_has_zero_trades():
    df = make_candles().iloc[:0]
    trades, summary = backtest_fixed_levels("TEST", df, one_signal)
    assert trades.empty
    assert summary["trades"] == 0


def test_summary_win_rate_and_expectancy():
    trades = pd.DataFrame({"net_pnl":[10.0, -5.0, 5.0]})
    summary = summarize_trades(trades, 1000)
    assert summary["win_rate_pct"] == pytest.approx(200/3)
    assert summary["expectancy_per_trade"] == pytest.approx(10/3)
    assert summary["net_pnl"] == 10.0


def test_quantity_is_risk_bounded():
    config = BacktestConfig(risk_per_trade=100, initial_capital=20_000)
    trades, _ = backtest_fixed_levels("TEST", make_candles(), one_signal, config)
    if not trades.empty:
        assert ((trades["entry"] - trades["stop_loss"]).abs() * trades["quantity"] <= 100).all()


def test_entry_at_or_after_cutoff_is_rejected():
    idx = pd.date_range("2026-10-12 14:35", periods=5, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "open": [100] * 5, "high": [101] * 5, "low": [99] * 5,
        "close": [100] * 5, "volume": [1000] * 5,
    }, index=idx)

    def signal_at_1440(history, i):
        if history.index[-1].strftime("%H:%M") == "14:40":
            return {"side": "LONG", "stop_loss": 98, "target": 104}
        return None

    trades, summary = backtest_fixed_levels("TEST", df, signal_at_1440)
    assert trades.empty
    assert summary["rejections"]["entry_cutoff"] >= 1


def test_position_is_forced_closed_at_1515():
    idx = pd.date_range("2026-10-12 14:35", "2026-10-12 15:10", freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "open": [100] * len(idx), "high": [101] * len(idx), "low": [99] * len(idx),
        "close": [100] * len(idx), "volume": [1000] * len(idx),
    }, index=idx)

    def first_signal(history, i):
        if i == 0:
            return {"side": "LONG", "stop_loss": 98, "target": 104}
        return None

    trades, summary = backtest_fixed_levels("TEST", df, first_signal)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "SESSION_SQUARE_OFF"
    assert "15:15:00" in trades.iloc[0]["exit_time"]
    assert trades.iloc[0]["entry_time"].endswith("14:40:00+05:30")


def test_net_risk_reward_below_minimum_is_rejected():
    idx = pd.date_range("2026-10-12 10:00", periods=4, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "open": [100] * 4, "high": [100.5] * 4, "low": [99.5] * 4,
        "close": [100] * 4, "volume": [1000] * 4,
    }, index=idx)

    def poor_rr(history, i):
        if i == 0:
            return {"side": "LONG", "stop_loss": 98, "target": 101}
        return None

    trades, summary = backtest_fixed_levels("TEST", df, poor_rr)
    assert trades.empty
    assert summary["rejections"]["net_rr_below_minimum"] >= 1
    diagnostics = summary["candidate_diagnostics"]
    assert diagnostics["diagnostic_candidates_recorded"] >= 1
    candidates = diagnostics["best_candidates_by_net_rr"]
    assert candidates
    candidate = candidates[0]
    assert candidate["planned_entry"] == pytest.approx(100.0)
    assert candidate["stop_loss"] == pytest.approx(98.0)
    assert candidate["target"] == pytest.approx(101.0)
    assert "brokerage" in candidate["initial_quantity_calculation"]["stop_exit_costs"]
    assert "slippage" in candidate["initial_quantity_calculation"]["target_exit_costs"]
    assert candidate["rejection_reason"] == "net_rr_below_minimum"


def test_cost_breakdown_is_reported_separately():
    trades, _ = backtest_fixed_levels("TEST", make_candles(), one_signal)
    if not trades.empty:
        for column in ["brokerage", "stt", "exchange_charges", "sebi_charges", "gst", "stamp_duty", "slippage"]:
            assert column in trades.columns
        row = trades.iloc[0]
        parts = sum(row[c] for c in ["brokerage", "stt", "exchange_charges", "sebi_charges", "gst", "stamp_duty", "ipft", "slippage"])
        assert row["estimated_costs"] == pytest.approx(parts)


def test_stop_loss_gap_uses_worse_open_price():
    idx = pd.date_range("2026-10-12 10:00", periods=4, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "open": [100, 100, 95, 95],
        "high": [100.5, 100.5, 96, 96],
        "low": [99.5, 99.5, 94, 94],
        "close": [100, 100, 95, 95],
        "volume": [1000] * 4,
    }, index=idx)

    def signal_on_first_bar(history, i):
        if i == 0:
            return {"side": "LONG", "stop_loss": 98, "target": 110}
        return None

    trades, _ = backtest_fixed_levels("TEST", df, signal_on_first_bar)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "STOP_LOSS"
    assert trades.iloc[0]["exit"] == pytest.approx(95.0)


def test_short_trade_pnl_is_calculated_correctly():
    idx = pd.date_range(
        "2026-10-12 10:00", periods=5, freq="5min", tz="Asia/Kolkata"
    )
    df = pd.DataFrame({
        "open":  [100, 100, 99, 98, 97],
        "high":  [101, 101, 100, 99, 98],
        "low":   [99, 99, 98, 97, 96],
        "close": [100, 100, 99, 98, 97],
    }, index=idx)

    def short_signal(history, i):
        if i == 0:
            return {"side": "SHORT", "stop_loss": 103, "target": 94}
        return None

    config = BacktestConfig(
        risk_per_trade=200,
        slippage_bps_per_side=0,
        minimum_net_risk_reward=1.0,
    )
    trades, _ = backtest_fixed_levels("TEST", df, short_signal, config)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["gross_pnl"] == pytest.approx(
        (trade["entry"] - trade["exit"]) * trade["quantity"]
    )
    assert trade["net_pnl"] == pytest.approx(
        trade["gross_pnl"] - trade["estimated_costs"]
    )


def test_stop_loss_wins_when_stop_and_target_hit_same_bar():
    idx = pd.date_range(
        "2026-10-12 10:00", periods=4, freq="5min", tz="Asia/Kolkata"
    )
    df = pd.DataFrame({
        "open":  [100, 100, 100, 100],
        "high":  [101, 101, 106, 101],
        "low":   [99, 99, 97, 99],
        "close": [100, 100, 101, 100],
    }, index=idx)

    def signal(history, i):
        if i == 0:
            return {"side": "LONG", "stop_loss": 98, "target": 105}
        return None

    trades, _ = backtest_fixed_levels("TEST", df, signal)

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "STOP_LOSS"


def test_net_pnl_reconciles_with_gross_pnl_and_costs():
    trades, _ = backtest_fixed_levels("TEST", make_candles(), one_signal)
    assert not trades.empty

    expected = trades["gross_pnl"] - trades["estimated_costs"]
    assert np.allclose(trades["net_pnl"], expected, atol=0.01)


def test_cash_only_notional_limit_is_respected():
    config = BacktestConfig(
        initial_capital=20_000,
        risk_per_trade=350,
        slippage_bps_per_side=0,
        minimum_net_risk_reward=1.0,
    )
    trades, _ = backtest_fixed_levels("TEST", make_candles(), one_signal, config)

    if not trades.empty:
        notional = trades["entry"] * trades["quantity"]
        assert (notional <= config.initial_capital).all()

