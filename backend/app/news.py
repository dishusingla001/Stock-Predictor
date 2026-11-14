"""Independent, bounded news retrieval and English headline sentiment.

Only a fixed Google News RSS host is fetched. Article links are returned to the
browser, never fetched by this service. Nothing is written to model/data files.
"""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import html
import logging
import re
from threading import Lock
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .schemas import NewsArticle, NewsCounts, NewsFeed
from .service import STOCK_NAMES, UnknownStock

LOGGER = logging.getLogger(__name__)
CACHE_TTL = timedelta(minutes=15)
STALE_LIMIT = timedelta(hours=1)
RETRY_DELAY = timedelta(minutes=1)
MAX_BYTES = 1_000_000
MAX_ARTICLES = 12
ALIASES = {
    "RELIANCE": ("Reliance Industries", "Reliance", "RIL"),
    "TCS": ("Tata Consultancy Services", "TCS"),
    "HDFCBANK": ("HDFC Bank", "HDFCBANK"),
    "INFY": ("Infosys", "INFY"),
    "ICICIBANK": ("ICICI Bank", "ICICIBANK"),
}


class NewsUnavailable(Exception):
    pass


def safe_url(value):
    try:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return None
        # Accessing port also validates malformed port values.
        parts.port
        return value.strip()
    except ValueError:
        return None


def clean_text(value):
    return " ".join(re.sub(r"<[^>]*>", "", html.unescape(value or "")).split())


def fetch_feed(symbol):
    query = '"' + STOCK_NAMES[symbol] + '" when:7d'
    url = "https://news.google.com/rss/search?" + urlencode(
        {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    )
    request = Request(url, headers={"User-Agent": "StockSight/0.1 (local news reader)",
                                    "Accept": "application/rss+xml, application/xml"})
    with urlopen(request, timeout=8) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("News feed exceeded its size limit.")
    return payload


def parse_feed(payload, symbol, now, analyzer):
    if len(payload) > MAX_BYTES or b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("Unsupported news feed.")
    root = ET.fromstring(payload)
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("The provider did not return an RSS feed.")
    articles, titles, links = [], set(), set()
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in ALIASES[symbol]) + r")(?!\w)"
    for item in root.findall("./channel/item")[:200]:
        title = clean_text(item.findtext("title"))
        source = clean_text(item.findtext("source"))
        # Publisher names can carry sentiment words; do not score them.
        if source and title.endswith(" - " + source):
            title = title[:-(len(source) + 3)].strip()
        link = safe_url(item.findtext("link") or "")
        if not title or len(title) > 500 or not source or len(source) > 200 or not link:
            continue
        if not re.search(pattern, title, flags=re.IGNORECASE):
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
            if published.tzinfo is None:
                continue
            published = published.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue
        if not now - timedelta(days=7) <= published <= now:
            continue
        title_key = re.sub(r"[^\w]+", " ", title.casefold()).strip()
        if title_key in titles or link in links:
            continue
        score = float(analyzer.polarity_scores(title)["compound"])
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        articles.append(NewsArticle(
            id=hashlib.sha256((title_key + link).encode()).hexdigest()[:20],
            title=title, source=source, url=link, published_at=published,
            sentiment=label, sentiment_score=score,
        ))
        titles.add(title_key)
        links.add(link)
    return sorted(articles, key=lambda article: article.published_at, reverse=True)[:MAX_ARTICLES]


class NewsService:
    def __init__(self, enabled=True, fetcher=fetch_feed, clock=None):
        self.enabled = enabled
        self.fetcher = fetcher
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cache = {}
        self.retry_after = {}
        self.locks = {symbol: Lock() for symbol in STOCK_NAMES}
        self.analyzer = None
        self.analyzer_lock = Lock()

    def _analyzer(self):
        with self.analyzer_lock:
            if self.analyzer is None:
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                except ImportError as exc:
                    raise NewsUnavailable("News sentiment requires the backend news dependency. Install backend/requirements.txt.") from exc
                self.analyzer = SentimentIntensityAnalyzer()
            return self.analyzer

    def _response(self, symbol, cached, is_cached, is_stale=False):
        fetched_at, articles = cached
        counts = {label: sum(article.sentiment == label for article in articles)
                  for label in ("positive", "neutral", "negative")}
        return NewsFeed(
            symbol=symbol, fetched_at=fetched_at, is_cached=is_cached, is_stale=is_stale,
            articles=articles, sentiment_counts=NewsCounts(**counts),
            detail="News refresh failed. Showing the last successful fetch." if is_stale else None,
        )

    def get(self, value):
        symbol = value.upper().removesuffix(".NS")
        if symbol not in STOCK_NAMES:
            raise UnknownStock("Unknown stock symbol: " + value)
        if not self.enabled:
            raise NewsUnavailable("News is disabled for this server.")
        with self.locks[symbol]:
            now = self.clock()
            cached = self.cache.get(symbol)
            age = now - cached[0] if cached else None
            if cached and age < CACHE_TTL:
                return self._response(symbol, cached, True)
            if now < self.retry_after.get(symbol, now):
                if cached and age <= STALE_LIMIT:
                    return self._response(symbol, cached, True, True)
                raise NewsUnavailable("News is temporarily unavailable. Please retry in a minute.")
            try:
                analyzer = self._analyzer()
                payload = self.fetcher(symbol)
                fetched_at = self.clock()
                articles = parse_feed(payload, symbol, fetched_at, analyzer)
            except (URLError, OSError, ValueError, ET.ParseError, NewsUnavailable) as exc:
                self.retry_after[symbol] = self.clock() + RETRY_DELAY
                LOGGER.warning("News fetch failed for %s (%s)", symbol, type(exc).__name__)
                if cached and self.clock() - cached[0] <= STALE_LIMIT:
                    return self._response(symbol, cached, True, True)
                raise NewsUnavailable("News is temporarily unavailable. Please retry in a minute.") from exc
            cached = (fetched_at, articles)
            self.cache[symbol] = cached
            self.retry_after.pop(symbol, None)
            return self._response(symbol, cached, False)
