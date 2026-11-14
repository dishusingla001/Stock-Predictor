"""Offline provider fixtures cover filtering, sentiment, cache, and HTTP failures."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from html import escape
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from fastapi.testclient import TestClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.news import MAX_BYTES, NewsService, NewsUnavailable, parse_feed
from backend.app.service import ServiceUnavailable, UnknownStock

NOW = datetime(2026, 11, 14, 10, tzinfo=timezone.utc)


def item(title="Reliance Industries reports excellent growth", link="https://news.google.com/rss/articles/example",
         published=None, source="Example News"):
    published = format_datetime(NOW - timedelta(hours=1)) if published is None else published
    return ("<item><title>" + escape(title + " - " + source) + "</title><link>" + escape(link)
            + "</link><pubDate>" + escape(published) + "</pubDate><source>" + escape(source) + "</source></item>")


def feed(*items):
    return ("<rss><channel>" + "".join(items) + "</channel></rss>").encode()


class NewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = SentimentIntensityAnalyzer()

    def parse(self, payload, symbol="RELIANCE"):
        return parse_feed(payload, symbol, NOW, self.analyzer)

    def test_real_vader_labels_and_excludes_publisher_from_scoring(self):
        articles = self.parse(feed(
            item("Reliance Industries reports excellent growth", link="https://example.com/positive"),
            item("Reliance Industries faces terrible fraud losses", link="https://example.com/negative"),
            item("Reliance Industries schedules annual meeting", link="https://example.com/neutral", source="Great Success"),
        ))
        by_title = {article.title: article for article in articles}
        self.assertEqual([article.sentiment for article in articles], ["positive", "negative", "neutral"])
        self.assertEqual(by_title["Reliance Industries schedules annual meeting"].sentiment_score, 0)
        self.assertNotIn("Great Success", articles[-1].title)

    def test_filters_bad_dates_old_future_unrelated_unsafe_and_duplicates(self):
        articles = self.parse(feed(
            item(),
            item(link="https://example.com/duplicate-title"),
            item("Reliance Industries old news", published=format_datetime(NOW - timedelta(days=8))),
            item("Reliance Industries future news", published=format_datetime(NOW + timedelta(minutes=1))),
            item("Reliance Industries missing date", published=""),
            item("Reliance Industries naive date", published="Sat, 14 Nov 2026 09:00:00"),
            item("Infosys reports results", link="https://example.com/unrelated"),
            item("Reliance Industries unsafe URL", link="javascript:alert(1)"),
            item("Reliance Industries credential URL", link="https://user:pass@example.com"),
            item("Reliance Industries malformed URL", link="https://[bad"),
            item("Reliance Industries invalid port", link="https://example.com:bad"),
        ))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].published_at.tzinfo, timezone.utc)

    def test_sort_limit_aliases_and_empty_feed(self):
        articles = self.parse(feed(*(item("Reliance Industries update " + str(index),
            link="https://example.com/" + str(index),
            published=format_datetime(NOW - timedelta(hours=index))) for index in range(20))))
        self.assertEqual(len(articles), 12)
        self.assertGreater(articles[0].published_at, articles[-1].published_at)
        self.assertEqual(len(self.parse(feed(item("TCS schedules annual meeting")), "TCS")), 1)
        self.assertEqual(len(self.parse(feed(item("STCS schedules annual meeting")), "TCS")), 0)
        self.assertEqual(self.parse(feed()), [])

    def test_rejects_non_rss_entities_and_oversized_payload(self):
        for payload in (b"<html/>", b'<!DOCTYPE rss><rss><channel/></rss>', b"x" * (MAX_BYTES + 1)):
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                self.parse(payload)

    def service(self, fetcher):
        service = NewsService(fetcher=fetcher, clock=lambda: self.now)
        self.now = NOW
        return service

    def test_cache_counts_symbol_normalization_and_cooldown(self):
        fetcher = Mock(return_value=feed(item()))
        service = self.service(fetcher)
        first = service.get("reliance.ns")
        self.assertFalse(first.is_cached)
        self.assertEqual(first.sentiment_counts.positive, 1)
        self.assertEqual(service.get("RELIANCE").fetched_at, NOW)
        self.assertTrue(service.get("RELIANCE").is_cached)
        self.assertEqual(fetcher.call_count, 1)
        self.now += timedelta(minutes=16)
        fetcher.side_effect = URLError("offline")
        stale = service.get("RELIANCE")
        self.assertTrue(stale.is_stale)
        self.assertEqual(stale.fetched_at, first.fetched_at)
        self.assertTrue(service.get("RELIANCE").is_stale)
        self.assertEqual(fetcher.call_count, 2)
        self.now += timedelta(hours=1)
        with self.assertRaises(NewsUnavailable):
            service.get("RELIANCE")
        count = fetcher.call_count
        with self.assertRaises(NewsUnavailable):
            service.get("RELIANCE")
        self.assertEqual(fetcher.call_count, count)

    def test_successful_empty_is_distinct_from_outage_and_disabled_does_not_fetch(self):
        fetcher = Mock(return_value=feed())
        service = self.service(fetcher)
        self.assertEqual(service.get("TCS").articles, [])
        disabled = NewsService(enabled=False, fetcher=fetcher)
        with self.assertRaises(NewsUnavailable):
            disabled.get("TCS")
        self.assertEqual(fetcher.call_count, 1)
        with self.assertRaises(UnknownStock):
            service.get("NOPE")
        with self.assertRaises(NewsUnavailable):
            self.service(Mock(side_effect=URLError("offline"))).get("TCS")

    def test_cache_recovers_after_provider_returns(self):
        fetcher = Mock(return_value=feed(item()))
        service = self.service(fetcher)
        service.get("RELIANCE")
        self.now += timedelta(minutes=16)
        fetcher.side_effect = URLError("offline")
        self.assertTrue(service.get("RELIANCE").is_stale)
        self.now += timedelta(minutes=1)
        fetcher.side_effect = None
        recovered = service.get("RELIANCE")
        self.assertFalse(recovered.is_stale)
        self.assertFalse(recovered.is_cached)
        self.assertEqual(recovered.fetched_at, self.now)

    def test_news_endpoint_works_independently_of_prediction_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(project_root=Path(directory))
            with patch("backend.app.main.StockService", side_effect=ServiceUnavailable("No model")), TestClient(create_app(settings)) as client:
                client.app.state.news = self.service(Mock(return_value=feed(item())))
                self.assertEqual(client.get("/health").status_code, 503)
                response = client.get("/api/v1/stocks/reliance.ns/news")
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["symbol"], "RELIANCE")
                self.assertEqual(data["articles"][0]["sentiment"], "positive")
                self.assertEqual(data["window_days"], 7)
                self.assertNotIn("up_probability", data)
                self.assertEqual(client.get("/api/v1/stocks/UNKNOWN/news").status_code, 404)
                self.assertEqual(client.get("/api/v1/stocks/bad-symbol/news").status_code, 422)
                client.app.state.news = self.service(Mock(side_effect=URLError("offline")))
                self.assertEqual(client.get("/api/v1/stocks/TCS/news").status_code, 503)
                self.assertEqual(client.get("/api/v1/stocks/RELIANCE/prediction").status_code, 503)
