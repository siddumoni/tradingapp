# Phase 9 — Extended historical validation (data-gap fix)

## Purpose
Run a multi-symbol, multi-strategy, slippage-sensitivity backtest using one configurable risk amount consistently across all scenarios. The runner never submits orders.

## Changes
- Accepts any finite positive `--risk-per-trade` amount; no arbitrary upper cap in the runner.
- Records the configured risk amount in each output CSV row.
- Fixes the over-strict session filter that rejected every supplied symbol because Yahoo Finance had small intraday candle gaps.
- Retains only sessions with 72–75 unique expected timestamps, starting at 09:15 and reaching at least 15:10 (some Yahoo sessions end at 15:15), with no out-of-session timestamps.
- Does not synthesize missing candles. Reports `sessions_kept_with_gaps` and `missing_bars_total` so gaps remain visible.
- Reuses one saved candle dataset per symbol across strategy/slippage scenarios.

## Install
Copy `src/multi_stock_validation.py` into the project's `src/` directory, replacing the previous Phase 9 version.

## Run from project root (PowerShell)
```powershell
python -m py_compile src/multi_stock_validation.py
python -m pytest -q
python -m src.multi_stock_validation --days 59 --risk-per-trade 350 --output-dir reports/backtests/phase9
```

The `350` amount is only an example. Use any finite positive INR amount appropriate to your plan. Keep the same cache/data and settings when comparing runs.

## Data limitation and caution
The uploaded Yahoo files contain 39 sessions per symbol and 72–74 rows per session. This patch accepts 72–74-row sessions when open/close timestamps are present, but missing bars are not reconstructed. A backtest can still be sensitive to the missing intervals, so review the gap counts and use the results as research rather than proof of profitability. Yahoo Finance 5-minute history does not provide a genuine 3–6 months in one run; longer testing requires a reliable longer-history source.

The runner does not submit orders and does not change the cash-only notional cap, minimum net R:R, time cutoffs, or advisory-only operation. Configured risk is a planned-loss budget, not a guarantee of realized maximum loss.
