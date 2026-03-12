import feedparser
import asyncio
from datetime import datetime
from email.utils import parsedate_to_datetime

RSS_FEEDS = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt":       "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
}

POLICY_KEYWORDS = [
    "regulation", "policy", "law", "bill", "sec", "cftc", "congress",
    "senate", "government", "ban", "legal tender", "etf", "approval",
    "executive order", "treasury", "irs", "tax", "compliance",
    "enforcement", "lawsuit", "ruling", "mica", "eu", "china", "japan",
    "korea", "india", "fca", "fsb", "g20", "imf", "world bank",
]


def _parse_entry(entry: dict, source: str) -> dict:
    published = None
    if hasattr(entry, "published"):
        try:
            published = parsedate_to_datetime(entry.published).isoformat()
        except Exception:
            published = entry.get("published", "")

    return {
        "title":     entry.get("title", ""),
        "summary":   entry.get("summary", "")[:300],
        "url":       entry.get("link", ""),
        "source":    source,
        "published": published,
    }


async def fetch_feed(name: str, url: str) -> list:
    loop = asyncio.get_event_loop()
    feed = await loop.run_in_executor(None, feedparser.parse, url)
    return [_parse_entry(e, name) for e in feed.entries[:15]]


async def get_news(limit: int = 40) -> list:
    tasks = [fetch_feed(name, url) for name, url in RSS_FEEDS.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    articles = []
    for r in results:
        if isinstance(r, list):
            articles.extend(r)

    articles.sort(key=lambda x: x["published"] or "", reverse=True)
    return articles[:limit]


async def get_policy_news(limit: int = 30) -> list:
    all_news = await get_news(limit=100)
    policy_articles = []
    for article in all_news:
        text = (article["title"] + " " + article["summary"]).lower()
        if any(kw in text for kw in POLICY_KEYWORDS):
            policy_articles.append(article)
    return policy_articles[:limit]
