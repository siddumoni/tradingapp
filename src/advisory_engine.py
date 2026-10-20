"""Phase 6 CLI: scan stocks, collect RSS headlines, and write advisory-only reports.

Example (Yahoo development data; not a verified live feed):
    python -m src.advisory_engine --symbols RELIANCE,HDFCBANK,ICICIBANK,SBIN,INFY

This command never places, modifies, or cancels orders.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from src.market_scanner import evaluate_symbol_bundle, rank_candidates
from src.news_events import DEFAULT_FEEDS, fetch_news, google_news_search_url
from src.risk_engine import load_phase2_settings

IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger("nse_advisory_assistant")


def load_watchlist(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    symbols = list(raw.get("nifty50", [])) + list(raw.get("additional_liquid_fo", []))
    return list(dict.fromkeys(str(x).strip().upper() for x in symbols if str(x).strip()))


def _fetch_nifty_15m(now: datetime) -> list[dict[str, Any]]:
    """Fetch NIFTY 50 index candles from Yahoo for research/regime context only."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is missing. Install project requirements before running the scanner.") from exc
    # Use recent sessions for indicator warm-up; the regime classifier needs more
    # than the handful of bars available shortly after the open.
    start = datetime.combine(now.date() - timedelta(days=7), dtime(9, 15), tzinfo=IST)
    frame = yf.Ticker("^NSEI").history(
        start=start.replace(tzinfo=None), end=(now + timedelta(minutes=1)).replace(tzinfo=None),
        interval="15m", auto_adjust=False, prepost=False, actions=False,
    )
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for idx, row in frame.iterrows():
        stamp = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=IST)
        else:
            stamp = stamp.astimezone(IST)
        rows.append({"date": stamp, "open": float(row["Open"]), "high": float(row["High"]), "low": float(row["Low"]), "close": float(row["Close"]), "volume": int(row.get("Volume", 0) or 0)})
    return rows


def _load_feeds(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return list(DEFAULT_FEEDS)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    feeds = raw.get("feeds", [])
    return [str(url).strip() for url in feeds if str(url).strip()] or list(DEFAULT_FEEDS)


def run_scan(
    *, symbols: list[str], now: datetime | None = None, max_symbols: int | None = None,
    watchlist_path: Path | None = None, news_feeds_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(IST)
    current = current.astimezone(IST) if current.tzinfo else current.replace(tzinfo=IST)
    if current.weekday() >= 5 or current.time() < dtime(9, 15) or current.time() > dtime(15, 30):
        return {"status": "MARKET_CLOSED", "as_of": current.isoformat(), "message": "NSE cash-market hours are normally 09:15–15:30 IST on weekdays. No intraday scan was run.", "candidates": [], "news": {"items": [], "errors": []}}

    try:
        from src.yahoo_data_fetcher import YahooFinanceMarketData
    except ImportError as exc:
        raise RuntimeError("Yahoo provider module is missing from src/yahoo_data_fetcher.py.") from exc
    provider = YahooFinanceMarketData()
    selected = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if max_symbols is not None:
        selected = selected[:max_symbols]
    if not selected and watchlist_path:
        selected = load_watchlist(watchlist_path)
        if max_symbols is not None:
            selected = selected[:max_symbols]
    if not selected:
        raise ValueError("No symbols supplied and no usable watchlist was found")

    news_result = fetch_news(_load_feeds(news_feeds_path), timeout_seconds=5.0)
    # Headlines are triage inputs only. Per-symbol company RSS queries are added for
    # the most promising technical candidates in a later iteration after this scan.
    try:
        index_15m = _fetch_nifty_15m(current)
    except Exception as exc:
        LOGGER.warning("Could not fetch NIFTY index candles: %s", exc)
        index_15m = []
        news_result.setdefault("errors", []).append(f"NIFTY index data unavailable: {exc}")

    risk_settings, charge_rates = load_phase2_settings()
    bundles: dict[str, dict[str, Any]] = {}
    fetch_errors: list[str] = []
    for symbol in selected:
        try:
            bundles[symbol] = provider.fetch_symbol_bundle(symbol, now=current)
        except Exception as exc:
            fetch_errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            LOGGER.warning("Skipping %s after data error: %s", symbol, exc)

    general_news = list(news_result.get("items", []))
    candidates = [
        evaluate_symbol_bundle(
            symbol, bundle, index_15m_candles=index_15m,
            news_items=general_news, margin_per_share=None,
            now=current, settings=risk_settings, charge_rates=charge_rates,
        )
        for symbol, bundle in bundles.items()
    ]

    # After the first technical pass, fetch company-specific RSS only for symbols
    # that actually produced a directional setup. This limits requests to the
    # small candidate set instead of querying every watchlist constituent.
    company_news_by_symbol: dict[str, list[dict[str, Any]]] = {}
    preliminary = [c for c in rank_candidates(candidates) if c.side is not None][:5]
    for candidate in preliminary:
        try:
            company_query = f'"{candidate.symbol}" stock NSE results earnings order'
            company_result = fetch_news(
                [google_news_search_url(company_query)], timeout_seconds=3.0, per_feed_limit=12
            )
            company_news_by_symbol[candidate.symbol] = company_result.get("items", [])
            news_result.setdefault("errors", []).extend(company_result.get("errors", []))
        except Exception as exc:
            company_news_by_symbol[candidate.symbol] = []
            news_result.setdefault("errors", []).append(f"Company news {candidate.symbol}: {exc}")

    if company_news_by_symbol:
        news_result["company_news_checked"] = sorted(company_news_by_symbol)
        news_result["items"] = general_news + [
            item for items in company_news_by_symbol.values() for item in items
        ]
        revised = []
        for candidate in candidates:
            if candidate.symbol not in company_news_by_symbol:
                revised.append(candidate)
                continue
            revised.append(evaluate_symbol_bundle(
                candidate.symbol, bundles[candidate.symbol], index_15m_candles=index_15m,
                news_items=general_news + company_news_by_symbol[candidate.symbol],
                margin_per_share=None, now=current, settings=risk_settings,
                charge_rates=charge_rates,
            ))
        candidates = revised

    ranked = rank_candidates(candidates)
    report = {
        "status": "RESEARCH_SCAN_ONLY",
        "as_of": current.isoformat(),
        "provider": "Yahoo Finance unofficial; not verified live exchange data",
        "advisory_only": True,
        "orders_placed_or_modified": False,
        "risk_settings": {
            "capital_inr": risk_settings.capital_inr,
            "risk_per_trade_inr": risk_settings.risk_per_trade_inr,
            "minimum_net_risk_reward": risk_settings.minimum_net_risk_reward,
            "require_margin_verification": risk_settings.require_margin_verification,
        },
        "symbols_requested": selected,
        "symbols_evaluated": len(candidates),
        "fetch_errors": fetch_errors,
        "news": news_result,
        "candidates": [item.to_dict() for item in ranked],
        "limitations": [
            "Yahoo Finance data is unofficial and may be delayed or incomplete; it is not approved for live trade decisions.",
            "Margin was not verified, so the risk engine is fail-closed and no candidate can be approved for entry.",
            "RSS keyword triage is not a sentiment model and does not replace original company filings or human verification.",
            "This report is advisory-only and never submits, modifies, or cancels orders.",
        ],
    }
    directory = output_dir or Path("reports")
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"phase6_scan_{current.strftime('%Y%m%d_%H%M%S')}.json"
    (directory / filename).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(directory / filename)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE advisory-only market scanner (research data mode)")
    parser.add_argument("--symbols", help="Comma-separated NSE symbols, e.g. RELIANCE,HDFCBANK,INFY")
    parser.add_argument("--max-symbols", type=int, default=10, help="Maximum symbols to scan (default: 10 to limit unofficial feed requests)")
    parser.add_argument("--watchlist", type=Path, default=Path("config/watchlist.yaml"))
    parser.add_argument("--news-feeds", type=Path, default=Path("config/news_feeds.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else load_watchlist(args.watchlist)
    report = run_scan(symbols=symbols, max_symbols=args.max_symbols, watchlist_path=args.watchlist, news_feeds_path=args.news_feeds, output_dir=args.output_dir)
    print(f"Status: {report['status']}")
    print(f"As of: {report['as_of']}")
    if report.get("report_path"):
        print(f"Report: {report['report_path']}")
    if report.get("message"):
        print(report["message"])
    for item in report.get("candidates", []):
        print(f"{item['symbol']:<14} {item['action']:<18} side={item['side'] or '-':<5} confidence={item['confidence']:>2} strategy={item['strategy'] or '-'}")
    print("No orders were placed, modified, or cancelled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
