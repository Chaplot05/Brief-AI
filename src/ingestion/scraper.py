"""
=============================================================
SCRAPER.PY — Article Scraping from URLs
=============================================================

WHAT THIS DOES:
    Takes a list of URLs (from data/urls.txt) and extracts the
    article text, title, and metadata from each one.

WHY WE NEED THIS:
    Our RAG system needs a knowledge base. We're building one
    about the Indian startup ecosystem by scraping real articles
    from TechCrunch India, YourStory, Inc42, etc.

LIBRARY: newspaper3k
    - Automatically extracts article text from any news URL
    - Handles HTML parsing, boilerplate removal (ads, nav bars)
    - Extracts title, authors, publish date
    - Alternative: BeautifulSoup (manual parsing), Scrapy (overkill)
    - newspaper3k is the sweet spot for article extraction

DATA FLOW:
    URLs (urls.txt) → scraper.py → JSON files (data/raw/)

PRODUCTION NOTE:
    In real companies, data ingestion is usually:
    - Scheduled (runs daily via cron/Airflow)
    - Has retry logic for failed URLs
    - Stores raw data before processing (data lake pattern)
    We keep it simple here but the structure supports scaling.

INTERVIEW QUESTION:
    "How did you handle data ingestion?"
    → "I built a scraping pipeline using newspaper3k that extracts
       article text from URLs, stores raw JSON with metadata, and
       handles failures gracefully with error logging."
=============================================================
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# newspaper3k for article extraction
from newspaper import Article

from src.config import RAW_DATA_DIR


def scrape_article(url: str) -> dict | None:
    """
    Scrape a single article from a URL.

    HOW IT WORKS:
        1. newspaper3k downloads the HTML page
        2. It parses the HTML to find the article body
        3. It removes boilerplate (navigation, ads, footers)
        4. It extracts: title, text, authors, publish date

    WHY RETURN A DICT?
        - Dicts are easy to serialize to JSON
        - Each article becomes a self-contained document
        - Metadata (title, URL, date) is crucial for citations later

    Args:
        url: The URL of the article to scrape

    Returns:
        dict with article data, or None if scraping failed
    """
    try:
        # Create an Article object — this is newspaper3k's main class
        article = Article(url)

        # download() fetches the HTML from the URL
        article.download()

        # parse() extracts the article content from the HTML
        # This is where the magic happens — it uses NLP heuristics
        # to identify the "main content" vs boilerplate
        article.parse()

        # Skip articles with too little text (likely scraping failures)
        if len(article.text) < 200:
            print(f"  ⚠ Skipped (too short): {url}")
            return None

        # Generate a unique ID for this article
        # We use MD5 hash of the URL — same URL always gets same ID
        # This prevents duplicate ingestion if you run the scraper twice
        article_id = hashlib.md5(url.encode()).hexdigest()[:12]

        return {
            "id": article_id,
            "url": url,
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": str(article.publish_date) if article.publish_date else None,
            "scraped_at": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"  ✗ Failed to scrape {url}: {e}")
        return None


def scrape_all_articles(urls_file: str | Path = None) -> list[dict]:
    """
    Scrape all articles from the URLs file.

    WHY SAVE RAW JSON?
        - "Data lake" pattern: always save raw data before processing
        - If your chunking strategy changes, you don't need to re-scrape
        - Debugging is easier when you can inspect raw data

    Args:
        urls_file: Path to file containing one URL per line.
                   Defaults to data/urls.txt

    Returns:
        List of article dicts that were successfully scraped
    """
    if urls_file is None:
        urls_file = Path(__file__).parent.parent.parent / "data" / "urls.txt"

    urls_file = Path(urls_file)
    if not urls_file.exists():
        raise FileNotFoundError(
            f"URLs file not found: {urls_file}\n"
            f"Create data/urls.txt with one article URL per line."
        )

    # Read URLs, skip empty lines and comments
    urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    print(f"📰 Scraping {len(urls)} articles...")
    articles = []

    for url in tqdm(urls, desc="Scraping"):
        article = scrape_article(url)
        if article:
            # Save each article as a separate JSON file
            output_path = RAW_DATA_DIR / f"{article['id']}.json"
            output_path.write_text(
                json.dumps(article, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            articles.append(article)
            print(f"  ✓ {article['title'][:60]}...")

    print(f"\n✅ Successfully scraped {len(articles)}/{len(urls)} articles")
    print(f"📁 Raw data saved to: {RAW_DATA_DIR}")
    return articles


def load_scraped_articles() -> list[dict]:
    """
    Load previously scraped articles from disk.

    WHY THIS FUNCTION?
        - You don't want to re-scrape every time you tweak chunking
        - This loads the raw JSON files saved by scrape_all_articles()
        - Separates "data collection" from "data processing"

    Returns:
        List of article dicts loaded from data/raw/
    """
    articles = []
    for json_file in sorted(RAW_DATA_DIR.glob("*.json")):
        article = json.loads(json_file.read_text(encoding="utf-8"))
        articles.append(article)

    print(f"📂 Loaded {len(articles)} articles from {RAW_DATA_DIR}")
    return articles
