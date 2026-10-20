"""Run Phase 4 advisory strategies against historical OHLCV candles.

Research only: this module never submits orders. Yahoo Finance is an unofficial
source; verify data coverage and candle timestamps before interpreting results.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Callable

import pandas as pd

from src.backtest import BacktestConfig, backtest_fixed_levels
from src.indicators import add_indicators
from src.strategies.levels import previous_session_levels  # noqa: F401
from src.strategies.market_regime import classify_market_regime
from src.strategies.opening_range import opening_range_signal
from src.strategies.signal_planner import plan_atr_levels
from src.strategies.vwap_pullback import vwap_pullback_signal
from src.strategies.vwap_trend import vwap_trend_signal
from src.yahoo_data_fetcher import YahooFinanceMarketData

LOGGER = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
SUPPORTED_STRATEGIES = ("opening_range", "vwap_trend", "vwap_pullback")


def _regime_is_filtered(regime: str, enabled: bool) -> bool:
    """Return whether the experimental non-trending-regime filter rejects a bar."""
    return enabled and regime in ("CHOPPY", "UNKNOWN")


def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Convert provider candle dictionaries to a timezone-aware OHLCV DataFrame."""
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(candles)
    if "date" not in frame:
        raise ValueError("Candle data must contain a 'date' field")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])



def filter_incomplete_sessions(
    candles: pd.DataFrame,
    *,
    session_start: str = "09:15",
    required_last_bar: str = "15:10",
    interval_minutes: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Keep only complete NSE sessions within the configured intraday window.

    The required window is inclusive of candle *start* timestamps from 09:15
    through 15:10 IST (the final candle covers 15:10–15:15). Any session with
    a missing expected timestamp, duplicate timestamp, or missing required OHLCV
    value is excluded as a whole. No synthetic candles are created.

    Naive timestamps are interpreted as IST; aware timestamps are converted to
    IST. The returned frame is restricted to the required window.
    """
    required_columns = ["open", "high", "low", "close", "volume"]
    if candles.empty:
        return candles.copy(), {
            "sessions_seen": 0,
            "sessions_kept": 0,
            "sessions_excluded": 0,
            "excluded_sessions": [],
            "required_bars_per_session": 72,
            "window_start_ist": session_start,
            "required_last_bar_ist": required_last_bar,
        }
    missing_columns = set(required_columns).difference(candles.columns)
    if missing_columns:
        raise ValueError(f"Missing candle columns: {sorted(missing_columns)}")
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("Candle DataFrame must have a DatetimeIndex")
    if candles.index.has_duplicates:
        duplicate_values = candles.index[candles.index.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate candle timestamps found: {duplicate_values[:5]}")

    frame = candles.copy()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(IST)
    else:
        frame.index = frame.index.tz_convert(IST)
    frame = frame.sort_index()

    start_hour, start_minute = map(int, session_start.split(":"))
    end_hour, end_minute = map(int, required_last_bar.split(":"))
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute
    if end_minutes < start_minutes or (end_minutes - start_minutes) % interval_minutes:
        raise ValueError("Invalid session window for the requested interval")

    required_times = [
        f"{minute // 60:02d}:{minute % 60:02d}"
        for minute in range(start_minutes, end_minutes + 1, interval_minutes)
    ]
    required_count = len(required_times)
    in_window = (
        (frame.index.time >= pd.Timestamp(session_start).time())
        & (frame.index.time <= pd.Timestamp(required_last_bar).time())
    )
    window = frame.loc[in_window].copy()
    if window.empty:
        return window, {
            "sessions_seen": 0,
            "sessions_kept": 0,
            "sessions_excluded": 0,
            "excluded_sessions": [],
            "required_bars_per_session": required_count,
            "window_start_ist": session_start,
            "required_last_bar_ist": required_last_bar,
        }

    kept_dates: list[str] = []
    excluded: list[dict] = []
    for day, group in window.groupby(window.index.date, sort=True):
        actual = set(group.index.strftime("%H:%M"))
        missing_times = [time for time in required_times if time not in actual]
        missing_values = group[required_columns].isna().any(axis=1)
        if missing_times or missing_values.any() or len(group) != required_count:
            excluded.append({
                "session": str(day),
                "actual_bars": int(len(group)),
                "required_bars": required_count,
                "missing_bars": missing_times,
                "rows_with_missing_ohlcv": int(missing_values.sum()),
                "reason": "missing_or_invalid_required_candles",
            })
        else:
            kept_dates.append(str(day))

    keep_mask = window.index.strftime("%Y-%m-%d").isin(kept_dates)
    filtered = window.loc[keep_mask].copy()
    diagnostics = {
        "sessions_seen": int(window.index.normalize().nunique()),
        "sessions_kept": len(kept_dates),
        "sessions_excluded": len(excluded),
        "excluded_sessions": excluded,
        "required_bars_per_session": required_count,
        "window_start_ist": session_start,
        "required_last_bar_ist": required_last_bar,
    }
    return filtered, diagnostics

def make_signal_callback(
    symbol: str,
    strategy_name: str,
    *,
    index_indicators: pd.DataFrame | None = None,
    target_atr_multiple: float = 2.5,
    reject_nontrending_regimes: bool = False,
) -> Callable[[pd.DataFrame, int], dict | None]:
    """Create the callback expected by backtest_fixed_levels.

    The callback sees only the candle history available at the signal close. If
    index data is provided, the regime uses index candles up to that timestamp;
    otherwise the traded symbol's own trend is used as a clearly imperfect proxy.
    ATR levels are anchored to the signal candle close. The backtester enters at
    the next candle open and skips gaps that put entry on the wrong side of a level.
    """
    if strategy_name not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy_name!r}; choose {SUPPORTED_STRATEGIES}")

    def callback(history: pd.DataFrame, _index: int) -> dict | None:
        if history.empty:
            return None
        timestamp = history.index[-1]
        if index_indicators is not None:
            regime_history = index_indicators.loc[index_indicators.index <= timestamp]
            regime = classify_market_regime(regime_history)
        else:
            regime = classify_market_regime(history)

        # Experimental filter: disabled by default to preserve baseline behavior.
        if _regime_is_filtered(regime, reject_nontrending_regimes):
            return None

        if strategy_name == "opening_range":
            signal = opening_range_signal(symbol, history, market_regime=regime)
        elif strategy_name == "vwap_trend":
            signal = vwap_trend_signal(symbol, history, market_regime=regime)
        else:
            signal = vwap_pullback_signal(symbol, history, market_regime=regime)

        if signal is None:
            return None
        row = history.iloc[-1]
        atr = float(row.get("atr_14", float("nan")))
        close = float(row["close"])
        if pd.isna(atr) or atr <= 0:
            return None
        levels = plan_atr_levels(
            signal.side, entry=close, atr=atr,
            stop_atr_multiple=1.0,
            # Define two research targets: 2 ATR and 2.5 ATR. Use the farther
            # target for this backtest because the engine evaluates minimum net
            # R:R after estimated costs. This is a hypothesis to test, not a
            # guarantee that a trade will pass the filter or be profitable.
            target_1_atr_multiple=min(2.0, target_atr_multiple),
            target_2_atr_multiple=target_atr_multiple,
        )
        return {
            "side": signal.side,
            "stop_loss": levels.stop_loss,
            "target": levels.target_2,
            "strategy": signal.strategy,
            "confidence": signal.confidence,
            "reason": signal.reason,
        }

    return callback


def backtest_strategy(
    symbol: str,
    candles: pd.DataFrame,
    strategy_name: str,
    config: BacktestConfig | None = None,
    index_candles: pd.DataFrame | None = None,
    target_atr_multiple: float = 2.5,
    reject_nontrending_regimes: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run one named strategy on an OHLCV DataFrame and return trades + metrics."""
    if candles.empty:
        raise ValueError("No candles supplied for backtesting")
    original_bars = len(candles)
    candles, data_quality = filter_incomplete_sessions(candles)
    if candles.empty:
        raise ValueError(
            "No complete trading sessions remain after data-quality filtering. "
            "Check the candle cache/provider coverage before backtesting."
        )
    indicators = add_indicators(candles)
    index_indicators = add_indicators(index_candles) if index_candles is not None and not index_candles.empty else None
    callback = make_signal_callback(
        symbol, strategy_name,
        index_indicators=index_indicators,
        target_atr_multiple=target_atr_multiple,
        reject_nontrending_regimes=reject_nontrending_regimes,
    )
    trades, summary = backtest_fixed_levels(symbol, indicators, callback, config)
    summary = dict(summary)
    summary.update({
        "symbol": symbol.upper(),
        "strategy": strategy_name,
        "target_atr_multiple": target_atr_multiple,
        "reject_nontrending_regimes": reject_nontrending_regimes,
        "slippage_bps_per_side": (
            float(config.slippage_bps_per_side) if config is not None else 5.0
        ),
        "data_start": str(indicators.index.min()),
        "data_end": str(indicators.index.max()),
        "bars": int(len(indicators)),
        "bars_before_session_filter": int(original_bars),
        "sessions_seen": data_quality["sessions_seen"],
        "sessions_kept": data_quality["sessions_kept"],
        "sessions_excluded": data_quality["sessions_excluded"],
        "excluded_sessions": data_quality["excluded_sessions"],
        "required_bars_per_session": data_quality["required_bars_per_session"],
        "session_filter_window": "09:15-15:10 IST (inclusive candle start times)",
        "regime_source": "index candles" if index_indicators is not None else "traded-symbol proxy",
        "data_warning": "Historical results are not predictive. Yahoo Finance data may be incomplete/delayed; costs are estimates and MIS margin is not assumed.",
    })
    if not trades.empty:
        trades["strategy"] = strategy_name
    return trades, summary


def run_yahoo_backtest(
    symbol: str,
    strategy_name: str,
    days: int = 30,
    output_dir: str | Path = "reports/backtests",
    risk_per_trade: float = 200.0,
    minimum_net_risk_reward: float = 1.5,
    target_atr_multiple: float = 2.5,
    reject_nontrending_regimes: bool = False,
    slippage_bps_per_side: float = 5.0,
) -> dict:
    """Fetch historical 5-minute Yahoo candles, run one strategy and save CSV/JSON."""
    if days < 5 or days > 59:
        raise ValueError("days must be between 5 and 59 for this intraday research workflow")
    now = datetime.now(IST)
    start = now - timedelta(days=days)
    provider = YahooFinanceMarketData()
    raw = provider.historical_candles(symbol, start, now, "5minute")
    frame = candles_to_frame(raw)
    if frame.empty:
        raise RuntimeError(f"No historical candles returned for {symbol}; no backtest was run.")
    if slippage_bps_per_side < 0 or slippage_bps_per_side > 100:
        raise ValueError("slippage_bps_per_side must be between 0 and 100")
    config = BacktestConfig(
        risk_per_trade=risk_per_trade,
        minimum_net_risk_reward=minimum_net_risk_reward,
        slippage_bps_per_side=slippage_bps_per_side,
    )
    if target_atr_multiple <= 1.0 or target_atr_multiple > 10.0:
        raise ValueError("target_atr_multiple must be greater than 1.0 and no more than 10.0")
    trades, summary = backtest_strategy(
        symbol, frame, strategy_name, config=config,
        target_atr_multiple=target_atr_multiple,
        reject_nontrending_regimes=reject_nontrending_regimes,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{symbol.upper()}_{strategy_name}_{stamp}"
    if trades.empty:
        # Preserve a useful CSV schema even when every candidate is rejected.
        # This makes "zero accepted trades" explicit rather than producing a
        # nearly blank 2-byte file with no column names.
        trades = pd.DataFrame(columns=[
            "symbol", "side", "entry_time", "exit_time", "entry", "exit",
            "stop_loss", "target", "quantity", "gross_pnl", "brokerage",
            "stt", "exchange_charges", "sebi_charges", "gst", "stamp_duty",
            "ipft", "slippage", "estimated_costs", "net_pnl", "exit_reason",
            "strategy",
        ])
    trades.to_csv(out / f"{stem}_trades.csv", index=False)
    # JSON does not support NaN/Infinity; represent an infinite profit factor as null.
    safe_summary = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in summary.items()
    }
    (out / f"{stem}_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(safe_summary, indent=2, allow_nan=False))
    print(f"Trade log: {out / f'{stem}_trades.csv'}")
    print(f"Summary:   {out / f'{stem}_summary.json'}")
    return safe_summary


def run_slippage_sensitivity(
    symbol: str, strategy_name: str, days: int = 30,
    output_dir: str | Path = "reports/backtests", risk_per_trade: float = 200.0,
    minimum_net_risk_reward: float = 1.5, target_atr_multiple: float = 2.5,
    reject_nontrending_regimes: bool = False,
    slippage_values: tuple[float, ...] = (2.0, 5.0, 8.0),
) -> pd.DataFrame:
    """Compare slippage assumptions using one identical downloaded candle dataset."""
    if days < 5 or days > 59:
        raise ValueError("days must be between 5 and 59 for this intraday research workflow")
    if not slippage_values or any(value < 0 or value > 100 for value in slippage_values):
        raise ValueError("slippage values must be between 0 and 100 bps per side")
    if target_atr_multiple <= 1.0 or target_atr_multiple > 10.0:
        raise ValueError("target_atr_multiple must be greater than 1.0 and no more than 10.0")
    now = datetime.now(IST)
    provider = YahooFinanceMarketData()
    raw = provider.historical_candles(symbol, now - timedelta(days=days), now, "5minute")
    frame = candles_to_frame(raw)
    if frame.empty:
        raise RuntimeError(f"No historical candles returned for {symbol}; no sensitivity test was run.")

    rows: list[dict] = []
    for bps in slippage_values:
        config = BacktestConfig(
            risk_per_trade=risk_per_trade,
            minimum_net_risk_reward=minimum_net_risk_reward,
            slippage_bps_per_side=float(bps),
        )
        trades, summary = backtest_strategy(
            symbol, frame, strategy_name, config=config,
            target_atr_multiple=target_atr_multiple,
            reject_nontrending_regimes=reject_nontrending_regimes,
        )
        rows.append({
            "symbol": symbol.upper(), "strategy": strategy_name,
            "slippage_bps_per_side": float(bps), "bars": len(frame),
            "trades": int(summary.get("trades", len(trades))),
            "win_rate_pct": summary.get("win_rate_pct"),
            "net_pnl": summary.get("net_pnl"),
            "profit_factor": summary.get("profit_factor"),
            "expectancy_per_trade": summary.get("expectancy_per_trade"),
            "max_drawdown": summary.get("max_drawdown"),
            "executed_trade_estimated_costs": (
                float(trades["estimated_costs"].astype(float).sum())
                if not trades.empty and "estimated_costs" in trades.columns else 0.0
            ),
            "ending_equity": summary.get("ending_equity"),
            "net_rr_rejections": (summary.get("rejections") or {}).get("net_rr_below_minimum"),
            "data_start": str(frame.index.min()), "data_end": str(frame.index.max()),
            "reject_nontrending_regimes": reject_nontrending_regimes,
        })
    result = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = out / f"{symbol.upper()}_{strategy_name}_slippage_sensitivity_{stamp}.csv"
    result.to_csv(report, index=False)
    print("Slippage sensitivity (same candle dataset for every row):")
    print(result.to_string(index=False))
    print(f"Comparison CSV: {report}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest an advisory-only intraday strategy on Yahoo Finance historical candles.")
    parser.add_argument("--symbol", required=True, help="NSE symbol, e.g. RELIANCE")
    parser.add_argument("--strategy", choices=SUPPORTED_STRATEGIES, required=True)
    parser.add_argument("--days", type=int, default=30, help="Calendar-day lookback (5–59)")
    parser.add_argument("--output-dir", default="reports/backtests")
    parser.add_argument("--risk-per-trade", type=float, default=200.0, help="Configurable planned loss budget per trade in INR; must be positive")
    parser.add_argument("--min-net-rr", type=float, default=1.5, help="Minimum net reward/risk after estimated charges and slippage")
    parser.add_argument("--target-atr-multiple", type=float, default=2.5, help="Target distance in ATR multiples (test values such as 3.5 or 4.0; does not relax the net R:R filter)")
    parser.add_argument("--filter-choppy-regimes", action="store_true", help="Experimental: reject entries when regime is CHOPPY or UNKNOWN")
    parser.add_argument("--slippage-bps", type=float, default=5.0, help="Assumed slippage per side in basis points (default: 5)")
    parser.add_argument("--slippage-sensitivity", action="store_true", help="Compare 2, 5, and 8 bps per side on identical candles")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not math.isfinite(args.risk_per_trade) or args.risk_per_trade <= 0:
        parser.error("--risk-per-trade must be a finite value greater than 0")
    if args.min_net_rr <= 0:
        parser.error("--min-net-rr must be positive")
    if args.target_atr_multiple <= 1.0 or args.target_atr_multiple > 10.0:
        parser.error("--target-atr-multiple must be greater than 1.0 and no more than 10.0")
    if args.slippage_bps < 0 or args.slippage_bps > 100:
        parser.error("--slippage-bps must be between 0 and 100")
    if args.slippage_sensitivity:
        run_slippage_sensitivity(
            args.symbol, args.strategy, args.days, args.output_dir,
            args.risk_per_trade, args.min_net_rr, args.target_atr_multiple,
            args.filter_choppy_regimes,
        )
    else:
        run_yahoo_backtest(
            args.symbol, args.strategy, args.days, args.output_dir,
            args.risk_per_trade, args.min_net_rr, args.target_atr_multiple,
            args.filter_choppy_regimes, args.slippage_bps,
        )


if __name__ == "__main__":
    main()
