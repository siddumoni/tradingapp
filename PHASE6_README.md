# Phase 6 — Market Scanner and Advisory Engine

## What this phase adds

- `src/market_scanner.py`: normalizes candles, removes unfinished bars, computes EMA/RSI/MACD/ATR/VWAP/volume indicators, classifies NIFTY regime when index candles are supplied, evaluates the existing Opening Range/VWAP strategies, creates ATR-based reference levels, and passes candidates through the existing cost/risk engine.
- `src/news_events.py`: dependency-free RSS collection, deduplication, and transparent keyword-based headline triage. This is not an AI sentiment model and is not a replacement for NSE/BSE filings.
- `src/advisory_engine.py`: command-line scanner, Yahoo research-data adapter, report output under `reports/`, and explicit limitations in every report.
- `config/news_feeds.yaml`: public RSS feed list.
- `tests/test_market_scanner.py` and `tests/test_news_events.py`: unit tests for candle handling, indicator calculation, closed-candle filtering, fail-closed behaviour, and RSS parsing.

## Current limitations (important)

1. The CLI currently uses unofficial Yahoo Finance data for development. It does **not** claim this is a verified real-time feed.
2. Margin verification is deliberately missing in this phase. The risk engine therefore rejects trade approval and sets the suggested quantity to zero. A candidate can show reference levels, but cannot be labelled an approved entry.
3. The RSS classifier is a keyword rule set, not an LLM or sentiment model. It can miss relevant stories and generate false positives.
4. Existing strategies are research candidates. The recent backtests did not establish a robust edge across stocks.
5. This phase is advisory-only. It contains no order placement, modification, or cancellation calls.

## Install dependencies

Use your existing project virtual environment. If the project requirements do not yet include these packages, install/update the project dependencies already used by prior phases (notably `pandas`, `ta`, `yfinance`, and `PyYAML`).

```powershell
python -m pip install pandas ta yfinance PyYAML
```

The new scanner computes its indicator set using pandas, but other existing modules and the Yahoo adapter may still require their earlier dependencies.

## Run unit tests

```powershell
python -m pytest tests/test_market_scanner.py tests/test_news_events.py -q
python -m pytest -q
```

## Run a small development scan

Run only during normal NSE cash-market hours on a weekday. Yahoo's latest candle is not guaranteed live and the result is research-only.

```powershell
python -m src.advisory_engine --symbols RELIANCE,HDFCBANK,ICICIBANK,SBIN,INFY --max-symbols 5
```

To scan the first 10 configured watchlist symbols:

```powershell
python -m src.advisory_engine --max-symbols 10
```

The command writes a timestamped JSON report to `reports/`. If the market is closed, it returns `MARKET_CLOSED` without fetching intraday data.

## Next implementation phase

Add a verified live-feed adapter, explicit quote freshness/spread checks, company-specific RSS and official exchange-announcement checks, and a validated margin-estimation source. Only after those gates exist should the program emit conditional trade advisories. Keep paper trading enabled until an out-of-sample test and forward paper-trading log justify further steps.
