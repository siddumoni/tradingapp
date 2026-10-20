"""Small dependency-free RSS collector and conservative news/event classifier.

This is a triage layer, not an AI sentiment model or a substitute for exchange
filings. Headlines that match risk keywords are surfaced for human verification.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
from email.utils import parsedate_to_datetime
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

USER_AGENT = "NSE-Advisory-Assistant/0.1 (RSS headline retrieval)"
DEFAULT_FEEDS = [
    "https://news.google.com/rss/search?q=Indian+stock+market+NSE+NIFTY+when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen",
    "https://news.google.com/rss/search?q=NSE+corporate+announcement+results+stocks+when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen",
    "https://news.google.com/rss/search?q=RBI+India+economy+crude+oil+rupee+when%3A1d&hl=en-IN&gl=IN&ceid=IN%3Aen",
]


def google_news_search_url(query: str, *, when: str = "1d") -> str:
    """Build a Google News RSS search URL for market or company-specific news."""
    q = f"{query} when:{when}"
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN%3Aen"


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def parse_rss(xml_bytes: bytes, *, source_url: str = "") -> list[dict[str, Any]]:
    """Parse RSS 2.0/Atom-ish feed entries into normalized headline records."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"RSS response is not valid XML: {exc}") from exc
    items: list[dict[str, Any]] = []
    # RSS 2.0 uses channel/item; Google News uses this format.
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        summary = (node.findtext("description") or "").strip()
        published = _parse_date(node.findtext("pubDate"))
        if title:
            items.append({"title": title, "url": link, "summary": summary, "published_at": published, "source_feed": source_url})
    # Minimal Atom fallback.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    if not items:
        for node in root.findall("atom:entry", ns):
            title = (node.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (node.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link_node = node.find("atom:link", ns)
            link = link_node.attrib.get("href", "") if link_node is not None else ""
            published = _parse_date(node.findtext("atom:published", default="", namespaces=ns))
            if title:
                items.append({"title": title, "url": link, "summary": summary, "published_at": published, "source_feed": source_url})
    return items


def fetch_rss_feed(url: str, *, timeout_seconds: float = 8.0, max_bytes: int = 2_000_000) -> list[dict[str, Any]]:
    """Fetch one RSS feed with a timeout and response-size limit."""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"RSS fetch failed for {url}: {exc}") from exc
    if len(payload) > max_bytes:
        raise RuntimeError(f"RSS response exceeded {max_bytes} bytes: {url}")
    if not payload:
        return []
    # Some providers omit or mislabel content-type; XML parsing is the validation.
    _ = content_type
    return parse_rss(payload, source_url=url)


def _normalise_title_key(title: str) -> str:
    """Build a title key that tolerates punctuation and publisher suffixes."""
    cleaned = _clean_news_text(title).casefold()
    # Google News commonly appends " - Publisher" to otherwise identical stories.
    if " - " in cleaned:
        cleaned = cleaned.rsplit(" - ", 1)[0]
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_news(
    feeds: Iterable[str] | None = None,
    *,
    timeout_seconds: float = 8.0,
    per_feed_limit: int = 30,
    max_age_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch fresh headlines, deduplicate syndicated copies and preserve failures.

    Only dated items inside ``max_age_hours`` are returned in ``items`` for use
    by the scanner. Stale and undated items are counted separately so unknown or
    old timestamps cannot silently influence current scan decisions.
    """
    if max_age_hours <= 0:
        raise ValueError("max_age_hours must be greater than zero")
    if per_feed_limit < 1:
        raise ValueError("per_feed_limit must be at least 1")

    urls = list(feeds or DEFAULT_FEEDS)
    all_items: list[dict[str, Any]] = []
    errors: list[str] = []
    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    reference_time = reference_time.astimezone(timezone.utc)
    fetched_at = reference_time.isoformat()
    seen: set[str] = set()
    stale_count = 0
    undated_count = 0
    duplicate_count = 0
    fetched_count = 0

    for url in urls:
        try:
            items = fetch_rss_feed(url, timeout_seconds=timeout_seconds)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        # Sort before applying the per-feed limit so old feed entries don't
        # crowd out newer stories if a provider returns items out of order.
        items = sorted(items, key=lambda x: str(x.get("published_at") or ""), reverse=True)
        for item in items[:per_feed_limit]:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            fetched_count += 1
            published = _parse_iso_datetime(item.get("published_at"))
            if published is None:
                undated_count += 1
                continue
            age_hours = (reference_time - published).total_seconds() / 3600
            if age_hours < -1:
                # More than one hour in the future is likely a bad timestamp.
                undated_count += 1
                continue
            if age_hours > max_age_hours:
                stale_count += 1
                continue

            key = _normalise_title_key(title)
            if not key:
                continue
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            normalised_item = dict(item)
            normalised_item["freshness_status"] = "FRESH"
            normalised_item["age_hours"] = round(max(0.0, age_hours), 2)
            all_items.append(normalised_item)

    all_items.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return {
        "fetched_at": fetched_at,
        "freshness_window_hours": max_age_hours,
        "feeds_attempted": len(urls),
        "feeds_failed": len(errors),
        "errors": errors,
        "items_fetched": fetched_count,
        "items_returned": len(all_items),
        "stale_items_excluded": stale_count,
        "undated_items_excluded": undated_count,
        "duplicate_items_excluded": duplicate_count,
        "items": all_items,
    }


def _clean_news_text(value: str) -> str:
    """Remove HTML markup and URLs before applying keyword rules."""
    text = unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def classify_headline(title: str, summary: str = "") -> dict[str, str]:
    """Return transparent keyword flags; this is not a claim of true sentiment."""
    # Use readable text only; RSS links often contain unrelated risk words in URLs.
    text = f"{_clean_news_text(title)} {_clean_news_text(summary)}".casefold()
    high = ("fraud", "raid", "default", "resigns", "resignation", "ban", "probe", "investigation", "downgrade", "profit warning", "earnings warning", "regulatory action", "penalty", "guidance cut", "cyberattack", "insolvency", "trading suspended")
    medium = ("results", "earnings", "merger", "acquisition", "order win", "large order", "policy", "rate decision", "inflation", "budget", "strike", "shutdown", "court", "approval", "rejection", "guidance", "dividend")
    if any(term in text for term in high):
        return {"risk_level": "HIGH", "method": "keyword_rule", "note": "Potentially material headline; verify original source and facts."}
    if any(term in text for term in medium):
        return {"risk_level": "MEDIUM", "method": "keyword_rule", "note": "Potential event headline; verify relevance and timing."}
    return {"risk_level": "LOW", "method": "keyword_rule", "note": "No configured high/medium risk keyword matched; this does not mean the news is harmless."}
