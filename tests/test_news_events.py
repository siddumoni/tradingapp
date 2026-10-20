from src.news_events import classify_headline, google_news_search_url, parse_rss


def test_parse_rss_normalizes_items():
    xml = b'''<?xml version="1.0"?><rss version="2.0"><channel><item><title>Company reports results</title><link>https://example.com/a</link><description>Revenue update</description><pubDate>Sat, 17 Oct 2026 08:00:00 GMT</pubDate></item></channel></rss>'''
    items = parse_rss(xml, source_url="https://feed.example/rss")
    assert len(items) == 1
    assert items[0]["title"] == "Company reports results"
    assert items[0]["url"] == "https://example.com/a"
    assert items[0]["published_at"].endswith("+00:00")


def test_classify_headline_uses_transparent_keyword_rules():
    assert classify_headline("Regulatory action and penalty announced")["risk_level"] == "HIGH"
    assert classify_headline("Company announces quarterly results")["risk_level"] == "MEDIUM"
    assert classify_headline("Market opens little changed")["risk_level"] == "LOW"


def test_google_news_url_encodes_query():
    url = google_news_search_url("RELIANCE Industries earnings")
    assert "news.google.com/rss/search" in url
    assert "RELIANCE" in url


def test_classify_headline_ignores_risk_words_only_in_link_markup():
    result = classify_headline(
        "Reliance shares trade steady",
        '<a href="https://example.com/raid-investigation-penalty">Read more</a>',
    )
    assert result["risk_level"] == "LOW"


from datetime import datetime, timezone
from src import news_events


def test_fetch_news_excludes_stale_and_undated_items(monkeypatch):
    now = datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
    items = [
        {"title": "Fresh company update", "url": "https://example.com/fresh", "published_at": "2026-10-19T07:30:00+00:00"},
        {"title": "Old market story", "url": "https://example.com/old", "published_at": "2026-10-17T07:00:00+00:00"},
        {"title": "Undated market story", "url": "https://example.com/unknown", "published_at": None},
    ]
    monkeypatch.setattr(news_events, "fetch_rss_feed", lambda *args, **kwargs: items)
    result = news_events.fetch_news(["https://feed.example/rss"], now=now, max_age_hours=24)
    assert [item["title"] for item in result["items"]] == ["Fresh company update"]
    assert result["stale_items_excluded"] == 1
    assert result["undated_items_excluded"] == 1
    assert result["items_returned"] == 1
    assert result["items"][0]["freshness_status"] == "FRESH"


def test_fetch_news_deduplicates_syndicated_titles(monkeypatch):
    now = datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
    items = [
        {"title": "Company announces major order - Publisher A", "url": "https://a.example/story", "published_at": "2026-10-19T07:30:00+00:00"},
        {"title": "Company announces major order - Publisher B", "url": "https://b.example/story", "published_at": "2026-10-19T07:35:00+00:00"},
        {"title": "Different company reports results", "url": "https://c.example/story", "published_at": "2026-10-19T07:20:00+00:00"},
    ]
    monkeypatch.setattr(news_events, "fetch_rss_feed", lambda *args, **kwargs: items)
    result = news_events.fetch_news(["https://feed.example/rss"], now=now)
    assert len(result["items"]) == 2
    assert result["duplicate_items_excluded"] == 1


def test_fetch_news_marks_freshness_and_sorts_newest_first(monkeypatch):
    now = datetime(2026, 10, 19, 8, 0, tzinfo=timezone.utc)
    items = [
        {"title": "Older fresh story", "url": "https://a.example/story", "published_at": "2026-10-19T06:00:00+00:00"},
        {"title": "Newer fresh story", "url": "https://b.example/story", "published_at": "2026-10-19T07:30:00+00:00"},
    ]
    monkeypatch.setattr(news_events, "fetch_rss_feed", lambda *args, **kwargs: items)
    result = news_events.fetch_news(["https://feed.example/rss"], now=now)
    assert [item["title"] for item in result["items"]] == ["Newer fresh story", "Older fresh story"]
    assert all(item["freshness_status"] == "FRESH" for item in result["items"])


def test_fetch_news_rejects_invalid_freshness_window():
    import pytest
    with pytest.raises(ValueError):
        news_events.fetch_news([], max_age_hours=0)
