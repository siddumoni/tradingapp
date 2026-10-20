"""Chronological train/test validation for two candidate NSE intraday strategies.

Research only: no orders are submitted. Uses saved OHLCV CSVs so every scenario
is reproducible and does not silently fetch different data between runs.

Place at src/chronological_validation.py and run from repository root:
  python -m src.chronological_validation --data-dir reports/backtests/candle_cache

Expected CSV names: INFY_30d_5minute.csv and HDFCBANK_30d_5minute.csv.
If your files are elsewhere, pass --data-dir with that directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd

from src.backtest import BacktestConfig, backtest_fixed_levels
from src.indicators import add_indicators
from src.strategy_backtest import (
    candles_to_frame,
    filter_incomplete_sessions,
    make_signal_callback,
)

LOGGER = logging.getLogger(__name__)
CANDIDATES = (("INFY", "vwap_pullback"), ("HDFCBANK", "vwap_trend"))
SLIPPAGE_BPS = (2.0, 5.0, 8.0)



def make_json_safe(value):
    """Replace non-finite floating-point values with JSON null recursively."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    # Handle NumPy scalar floats without requiring a NumPy import here.
    if hasattr(value, "item"):
        try:
            scalar = value.item()
        except (ValueError, TypeError):
            return value
        if isinstance(scalar, float) and not math.isfinite(scalar):
            return None
        if scalar is not value:
            return make_json_safe(scalar)
    return value


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candle CSV: {path}")
    frame = pd.read_csv(path)
    frame = candles_to_frame(frame.to_dict("records"))
    if frame.empty:
        raise ValueError(f"No valid OHLC candles in {path}")
    # Do not silently drop duplicate timestamps: the quality filter must detect
    # them and fail loudly, rather than changing the input without reporting it.
    return frame.sort_index()


def split_by_session(
    frame: pd.DataFrame, train_fraction: float
) -> tuple[pd.Timestamp, pd.Timestamp, list, list]:
    """Split only the already quality-filtered, complete sessions chronologically."""
    sessions = sorted(pd.Index(frame.index.date).unique())
    if len(sessions) < 10:
        raise ValueError(
            "Need at least 10 complete sessions after data-quality filtering; "
            f"found {len(sessions)}"
        )
    split_idx = int(len(sessions) * train_fraction)
    split_idx = min(max(split_idx, 1), len(sessions) - 1)
    train_sessions = sessions[:split_idx]
    test_sessions = sessions[split_idx:]
    tz = frame.index.tz
    test_start = pd.Timestamp(test_sessions[0]).tz_localize(tz)
    test_end = pd.Timestamp(test_sessions[-1]).tz_localize(tz) + pd.Timedelta(days=1)
    return test_start, test_end, train_sessions, test_sessions


def run_one(
    symbol: str,
    strategy: str,
    frame: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    slippage: float,
    risk: float,
    min_rr: float,
    quality_report: dict,
    train_sessions: list,
    test_sessions: list,
) -> dict:
    # `frame` contains only complete sessions. Compute indicators over the
    # quality-filtered timeline so incomplete days cannot contribute signals or
    # contaminate indicator state. Earlier complete sessions still provide warm-up.
    full_indicators = add_indicators(frame)
    test_indicators = full_indicators.loc[
        (full_indicators.index >= test_start) & (full_indicators.index < test_end)
    ].copy()
    if test_indicators.empty:
        raise ValueError("Test partition contains no candles")
    # Callback sees only test-period bars; pre-test data is used for indicator warm-up,
    # not for test-period trade entries. This avoids training-period trades leaking in.
    callback = make_signal_callback(symbol, strategy)
    config = BacktestConfig(
        risk_per_trade=risk,
        minimum_net_risk_reward=min_rr,
        slippage_bps_per_side=slippage,
    )
    trades, summary = backtest_fixed_levels(symbol, test_indicators, callback, config)
    result = dict(summary)
    result.update({
        "symbol": symbol,
        "strategy": strategy,
        "partition": "chronological_test",
        "slippage_bps_per_side": slippage,
        "test_start": str(test_indicators.index.min()),
        "test_end": str(test_indicators.index.max()),
        "test_bars": int(len(test_indicators)),
        "test_sessions": int(pd.Index(test_indicators.index.date).nunique()),
        "train_sessions": len(train_sessions),
        "total_complete_sessions": int(pd.Index(frame.index.date).nunique()),
        "sessions_seen_before_filter": quality_report["sessions_seen"],
        "sessions_kept": quality_report["sessions_kept"],
        "sessions_excluded": quality_report["sessions_excluded"],
        "excluded_sessions": quality_report["excluded_sessions"],
        "required_bars_per_session": quality_report["required_bars_per_session"],
        "session_filter_window": "09:15-15:10 IST (inclusive candle start times)",
        "bars_before_session_filter": int(quality_report.get("bars_before_filter", len(frame))),
        "bars_after_session_filter": int(len(frame)),
        "data_source": "saved Yahoo Finance 5-minute CSV; incomplete sessions excluded",
        "warning": "Small test samples are inconclusive; not a live-trading recommendation.",
    })
    if not trades.empty:
        result["trade_log"] = trades.to_dict("records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological out-of-sample validation (research only).")
    parser.add_argument("--data-dir", default="reports/backtests/candle_cache", help="Directory containing saved 5-minute CSVs")
    parser.add_argument("--output-dir", default="reports/backtests/chronological_validation")
    parser.add_argument("--days-label", default="30d", help="CSV filename suffix, e.g. 30d")
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Fraction of sessions before the test partition; default 0.70")
    parser.add_argument("--risk-per-trade", type=float, default=200.0)
    parser.add_argument("--min-net-rr", type=float, default=1.5)
    args = parser.parse_args()
    if not 0.50 <= args.train_fraction <= 0.85:
        parser.error("--train-fraction must be between 0.50 and 0.85")
    if not 0 < args.risk_per_trade <= 400:
        parser.error("--risk-per-trade must be > 0 and <= 400")
    if args.min_net_rr <= 0:
        parser.error("--min-net-rr must be positive")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for symbol, strategy in CANDIDATES:
        path = data_dir / f"{symbol}_{args.days_label}_5minute.csv"
        try:
            raw_frame = load_csv(path)
            filtered_frame, quality_report = filter_incomplete_sessions(raw_frame)
            quality_report["bars_before_filter"] = int(len(raw_frame))
            if filtered_frame.empty:
                raise ValueError(
                    "No complete sessions remain after data-quality filtering. "
                    f"Quality report: {quality_report}"
                )
            frame = filtered_frame
            test_start, test_end, train_sessions, test_sessions = split_by_session(
                frame, args.train_fraction
            )
            LOGGER.info(
                "%s: sessions seen=%d, kept=%d, excluded=%d; train=%d; test=%d (%s to %s)",
                symbol, quality_report["sessions_seen"], quality_report["sessions_kept"],
                quality_report["sessions_excluded"], len(train_sessions), len(test_sessions),
                test_sessions[0], test_sessions[-1],
            )
            if quality_report["sessions_excluded"]:
                LOGGER.warning("%s excluded sessions: %s", symbol, quality_report["excluded_sessions"])
            for slippage in SLIPPAGE_BPS:
                try:
                    row = run_one(
                        symbol, strategy, frame, test_start, test_end, slippage,
                        args.risk_per_trade, args.min_net_rr, quality_report,
                        train_sessions, test_sessions,
                    )
                    row["total_sessions"] = len(train_sessions) + len(test_sessions)
                    row["source_csv"] = str(path)
                    # Infinite profit factor (e.g. one winning trade and no losses)
                    # is undefined for strict JSON; write it as null instead of
                    # failing the scenario after its metrics have already been computed.
                    row = make_json_safe(row)
                    # Keep CSV summary compact; full trade details are in the JSON report.
                    rows.append({k: v for k, v in row.items() if k != "trade_log" and k != "rejections"})
                    (output_dir / f"{symbol}_{strategy}_{int(slippage)}bps_test.json").write_text(
                        json.dumps(row, indent=2, default=str, allow_nan=False), encoding="utf-8")
                except Exception as exc:
                    LOGGER.exception("Scenario failed for %s %s at %.1f bps", symbol, strategy, slippage)
                    rows.append({"symbol": symbol, "strategy": strategy,
                                 "slippage_bps_per_side": slippage, "error": str(exc),
                                 "source_csv": str(path)})
        except Exception as exc:
            LOGGER.exception("Could not validate %s", symbol)
            rows.append({"symbol": symbol, "strategy": strategy, "error": str(exc), "source_csv": str(path)})

    result = pd.DataFrame(rows)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"chronological_validation_{stamp}.csv"
    result.to_csv(csv_path, index=False)
    print(result.to_string(index=False))
    print(f"\nSummary CSV: {csv_path}")
    print(f"Detailed JSON files: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
