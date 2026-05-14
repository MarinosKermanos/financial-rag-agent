import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = os.getenv("NEWS_API_URL")

DEFAULT_QUERIES = [
    "forex EUR USD",
    "Federal Reserve interest rates",
    "oil price OPEC",
    "S&P 500 stock market",
    "crypto Bitcoin",
]

@dataclass
class Article:
    title: str
    content: str
    url: str
    published_at: str
    source: str


def fetch_articles(query: str, page_size: int = 10) -> list[Article]:
    """Fetch articles from NewsAPI for a given query."""
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    articles = []
    for item in data.get("articles", []):
        # NewsAPI often returns [Removed] content on free tier — skip those
        content = item.get("content") or item.get("description") or ""
        if not content or "[Removed]" in content:
            continue

        articles.append(Article(
            title=item.get("title", ""),
            content=content,
            url=item.get("url", ""),
            published_at=item.get("publishedAt", ""),
            source=item.get("source", {}).get("name", "Unknown"),
        ))

    return articles


def fetch_all_articles() -> list[Article]:
    """Fetch articles for all default financial queries."""
    all_articles = []
    seen_urls = set()  # deduplicate by URL

    for query in DEFAULT_QUERIES:
        print(f"Fetching: {query}")
        articles = fetch_articles(query)
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                all_articles.append(article)

    print(f"Total unique articles fetched: {len(all_articles)}")
    return all_articles


if __name__ == "__main__":
    articles = fetch_all_articles()
    for a in articles[:3]:
        print(f"\n--- {a.title} ---")
        print(f"Source: {a.source} | Published: {a.published_at}")
        print(a.content[:200])