"""
=============================================================
SCRAPE_REAL.PY — Scrape REAL articles from Wikipedia + news sites
=============================================================

WHY WIKIPEDIA?
    - Server-rendered HTML (no JavaScript needed)
    - newspaper3k works perfectly with Wikipedia
    - Factual, well-structured content
    - Always available, no paywalls
    
    We ALSO try some news sites that work without JS.

THIS REPLACES seed_data.py — now our knowledge base is REAL scraped data,
not content we wrote ourselves.
=============================================================
"""

import sys
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR

# Wikipedia articles about Indian startups — these ALWAYS work
URLS = [
    # Major Indian Startups
    "https://en.wikipedia.org/wiki/Zerodha",
    "https://en.wikipedia.org/wiki/Razorpay",
    "https://en.wikipedia.org/wiki/Flipkart",
    "https://en.wikipedia.org/wiki/PhonePe",
    "https://en.wikipedia.org/wiki/CRED_(company)",
    "https://en.wikipedia.org/wiki/Meesho",
    "https://en.wikipedia.org/wiki/Ola_Cabs",
    "https://en.wikipedia.org/wiki/Ola_Electric",
    "https://en.wikipedia.org/wiki/Nykaa",
    "https://en.wikipedia.org/wiki/Groww",
    "https://en.wikipedia.org/wiki/Byju%27s",
    "https://en.wikipedia.org/wiki/Paytm",
    "https://en.wikipedia.org/wiki/Swiggy",
    "https://en.wikipedia.org/wiki/Zomato",
    "https://en.wikipedia.org/wiki/Zepto_(company)",
    "https://en.wikipedia.org/wiki/Delhivery",
    
    # Ecosystem / Infra
    "https://en.wikipedia.org/wiki/Unified_Payments_Interface",
    "https://en.wikipedia.org/wiki/Startup_India",
    "https://en.wikipedia.org/wiki/Zoho_Corporation",
    "https://en.wikipedia.org/wiki/Freshworks",
    "https://en.wikipedia.org/wiki/Infosys",
    "https://en.wikipedia.org/wiki/Lenskart",
    "https://en.wikipedia.org/wiki/Dream11",
    "https://en.wikipedia.org/wiki/Pine_Labs",
    "https://en.wikipedia.org/wiki/Unacademy",
    "https://en.wikipedia.org/wiki/Dunzo",
    "https://en.wikipedia.org/wiki/Urban_Company",
    "https://en.wikipedia.org/wiki/ShareChat",
    "https://en.wikipedia.org/wiki/Postman_(software)",
    "https://en.wikipedia.org/wiki/BharatPe",
]


def scrape_wikipedia(url: str) -> dict | None:
    """Scrape a Wikipedia article using newspaper3k."""
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()

        if len(article.text) < 300:
            print(f"  ! Too short ({len(article.text)} chars): {url}")
            return None

        article_id = "wiki_" + hashlib.md5(url.encode()).hexdigest()[:10]
        return {
            "id": article_id,
            "title": article.title.replace(" - Wikipedia", ""),
            "url": url,
            "text": article.text,
            "authors": ["Wikipedia"],
            "publish_date": datetime.now().isoformat(),
            "source_type": "wikipedia",
            "char_count": len(article.text),
        }
    except Exception as e:
        print(f"  X Failed: {url} — {e}")
        return None


def main():
    print("=" * 60)
    print("Scraping REAL articles from Wikipedia...")
    print("=" * 60)

    # Clear old seed data
    old_files = list(RAW_DATA_DIR.glob("*.json"))
    if old_files:
        print(f"\nRemoving {len(old_files)} old seed files...")
        for f in old_files:
            f.unlink()

    success = 0
    for i, url in enumerate(URLS, 1):
        print(f"\n[{i}/{len(URLS)}] {url.split('/')[-1]}...")
        article = scrape_wikipedia(url)
        if article:
            path = RAW_DATA_DIR / f"{article['id']}.json"
            path.write_text(
                json.dumps(article, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"  OK — {article['title'][:50]} ({article['char_count']} chars)")
            success += 1

    print(f"\n{'=' * 60}")
    print(f"Scraped {success}/{len(URLS)} articles successfully")
    print(f"Saved to: {RAW_DATA_DIR}")
    print(f"{'=' * 60}")
    print(f"\nNext: run 'python scripts/ingest.py' to chunk and embed")


if __name__ == "__main__":
    main()
