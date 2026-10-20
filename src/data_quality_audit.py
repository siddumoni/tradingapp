"""Audit saved NSE 5-minute OHLCV CSVs for the project's trading window.

Place at src/data_quality_audit.py and run from the repository root:
  python -m src.data_quality_audit --data-dir reports/backtests/candle_cache

The required trading window is 09:15 through 15:10 inclusive. With start-
 timestamped 5-minute candles, the 15:10 candle covers 15:10-15:15, matching
the project's 15:15 square-off. Missing bars after 15:10 are reported separately
and do not make a session incomplete for this strategy's execution window.
The script is read-only and does not fetch data or place orders.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

SYMBOLS = ("INFY", "HDFCBANK")
OHLC = ("Open", "High", "Low", "Close")


def parse_clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def expected_grid(day, start: str, end: str, tz) -> pd.DatetimeIndex:
    start_text, end_text = f"{day} {start}", f"{day} {end}"
    if tz is not None:
        return pd.date_range(start=pd.Timestamp(start_text, tz=tz), end=pd.Timestamp(end_text, tz=tz), freq="5min")
    return pd.date_range(start_text, end_text, freq="5min")


def load_candles(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    time_col = next((c for c in ("Datetime", "datetime", "Date", "date", "Timestamp", "timestamp") if c in raw.columns), None)
    if time_col is None and len(raw.columns) and raw.columns[0].lower().startswith("unnamed"):
        time_col = raw.columns[0]
    if time_col is None:
        raise ValueError("No recognizable timestamp column (Datetime/Date/Timestamp) found")
    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw.dropna(subset=[time_col]).set_index(time_col).sort_index()
    rename = {c: c.title() for c in raw.columns if c.lower() in {"open", "high", "low", "close", "volume"}}
    raw = raw.rename(columns=rename)
    missing = [c for c in OHLC if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    for col in (*OHLC, *(("Volume",) if "Volume" in raw.columns else ())):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def audit_symbol(symbol: str, path: Path, required_until: str, session_end: str) -> tuple[dict, list[dict]]:
    frame = load_candles(path)
    idx = frame.index
    duplicated = int(idx.duplicated(keep=False).sum())
    invalid_rows = frame[list(OHLC)].isna().any(axis=1)
    nonpositive = (frame[list(OHLC)] <= 0).any(axis=1)
    bad_ohlc = (
        (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1))
        | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))
        | (frame["High"] < frame["Low"])
    )
    volume_bad = pd.Series(False, index=frame.index)
    if "Volume" in frame.columns:
        volume_bad = frame["Volume"].isna() | (frame["Volume"] < 0)

    session_rows: list[dict] = []
    for day, group in frame.groupby(frame.index.date, sort=True):
        # Standard session bars are timestamped by their starting time.
        session = group.between_time("09:15", session_end)
        observed = pd.DatetimeIndex(session.index).unique()
        required = expected_grid(day, "09:15", required_until, idx.tz)
        full_session = expected_grid(day, "09:15", session_end, idx.tz)
        missing_required = required.difference(observed)
        missing_after_cutoff = full_session.difference(required).difference(observed)
        first_bar = session.index.min() if not session.empty else None
        last_bar = session.index.max() if not session.empty else None
        starts_at_open = bool(first_bar is not None and first_bar.strftime("%H:%M") == "09:15")
        has_cutoff_bar = bool(required[-1] in observed)
        complete_required_window = bool(starts_at_open and has_cutoff_bar and len(missing_required) == 0)
        outside = int(((group.index.time < pd.Timestamp("09:15").time()) | (group.index.time > pd.Timestamp(session_end).time())).sum())
        session_rows.append({
            "symbol": symbol,
            "session_date": str(day),
            "bars_in_required_window": int(session.index.to_series().between(required[0], required[-1]).sum()) if not session.empty else 0,
            "expected_bars_through_squareoff": int(len(required)),
            "missing_bars_through_squareoff": int(len(missing_required)),
            "missing_required_times": ";".join(x.strftime("%H:%M") for x in missing_required),
            "first_bar": str(first_bar) if first_bar is not None else "",
            "last_bar": str(last_bar) if last_bar is not None else "",
            "starts_at_0915": starts_at_open,
            "has_1510_squareoff_bar": has_cutoff_bar,
            "complete_for_trading_window": complete_required_window,
            "missing_bars_after_squareoff_window": int(len(missing_after_cutoff)),
            "missing_after_cutoff_times": ";".join(x.strftime("%H:%M") for x in missing_after_cutoff),
            "outside_regular_hours_rows": outside,
        })

    complete_count = sum(bool(row["complete_for_trading_window"]) for row in session_rows)
    summary = {
        "symbol": symbol,
        "source_file": str(path),
        "rows": int(len(frame)),
        "first_timestamp": str(idx.min()) if len(idx) else "",
        "last_timestamp": str(idx.max()) if len(idx) else "",
        "distinct_sessions": int(pd.Index(idx.date).nunique()),
        "duplicate_timestamp_rows": duplicated,
        "rows_with_missing_ohlc": int(invalid_rows.sum()),
        "rows_with_nonpositive_ohlc": int(nonpositive.sum()),
        "rows_with_invalid_ohlc_relationships": int(bad_ohlc.sum()),
        "rows_with_invalid_volume": int(volume_bad.sum()),
        "sessions_complete_through_1510": int(complete_count),
        "sessions_incomplete_through_1510": int(len(session_rows) - complete_count),
        "sessions_with_internal_required_gaps": int(sum(row["missing_bars_through_squareoff"] > 0 for row in session_rows)),
        "sessions_missing_1510_bar": int(sum(not row["has_1510_squareoff_bar"] for row in session_rows)),
        "assessment": "ERRORS_FOUND" if any([duplicated, invalid_rows.any(), nonpositive.any(), bad_ohlc.any(), volume_bad.any()]) else ("WINDOW_COMPLETE" if complete_count == len(session_rows) else "REVIEW_REQUIRED_WINDOW_GAPS"),
    }
    return summary, session_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of saved NSE 5-minute candles through the strategy square-off window")
    parser.add_argument("--data-dir", default="reports/backtests/candle_cache")
    parser.add_argument("--output-dir", default="reports/backtests/data_quality_audit")
    parser.add_argument("--days-label", default="30d")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--required-until", default="15:10", help="Last required candle start time; default 15:10 for 15:15 square-off")
    parser.add_argument("--session-end", default="15:25", help="Last expected regular-session candle start time")
    args = parser.parse_args()
    for value_name, value in (("required-until", args.required_until), ("session-end", args.session_end)):
        try:
            parsed = pd.to_datetime(value, format="%H:%M", errors="raise")
        except Exception as exc:
            parser.error(f"--{value_name} must be HH:MM: {exc}")
    if args.required_until > args.session_end:
        parser.error("--required-until cannot be later than --session-end")

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, sessions = [], []
    for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        path = data_dir / f"{symbol}_{args.days_label}_5minute.csv"
        try:
            summary, rows = audit_symbol(symbol, path, args.required_until, args.session_end)
            summaries.append(summary)
            sessions.extend(rows)
        except Exception as exc:
            summaries.append({"symbol": symbol, "source_file": str(path), "assessment": "ERROR", "error": str(exc)})
    summary_df = pd.DataFrame(summaries)
    sessions_df = pd.DataFrame(sessions)
    summary_path = output_dir / "data_quality_summary.csv"
    sessions_path = output_dir / "data_quality_sessions.csv"
    summary_df.to_csv(summary_path, index=False)
    sessions_df.to_csv(sessions_path, index=False)
    print("SYMBOL SUMMARY")
    print(summary_df.to_string(index=False))
    print("\nSESSION DETAIL (first 30 rows)")
    print(sessions_df.head(30).to_string(index=False))
    print(f"\nSummary: {summary_path}\nSessions: {sessions_path}")


if __name__ == "__main__":
    main()
