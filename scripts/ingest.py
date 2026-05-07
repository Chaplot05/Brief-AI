"""
=============================================================
INGEST.PY — Run the Full Ingestion Pipeline
=============================================================

USAGE:
    python scripts/ingest.py

WHAT IT DOES:
    1. Scrapes articles from data/urls.txt
    2. Chunks the article text
    3. Embeds chunks and stores in ChromaDB

RUN THIS ONCE to build your knowledge base.
After that, you can query without re-running ingestion.
=============================================================
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work
# This is needed because we're running from scripts/ directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.scraper import scrape_all_articles, load_scraped_articles
from src.ingestion.chunker import chunk_articles
from src.ingestion.embedder import embed_and_store, get_collection_stats


def main():
    print("=" * 60)
    print("🚀 CORRECTIVE RAG — INGESTION PIPELINE")
    print("=" * 60)

    # ── Step 1: Load Articles ──────────────────────────────
    print("\nSTEP 1: Loading articles...")
    articles = load_scraped_articles()

    if not articles:
        print("No articles found. Run 'python scripts/seed_data.py' first!")
        sys.exit(1)

    # ── Step 2: Chunk Articles ──────────────────────────────
    print("\n📦 STEP 2: Chunking articles...")
    chunks = chunk_articles(articles)

    # ── Step 3: Embed & Store ───────────────────────────────
    print("\n🔢 STEP 3: Embedding and storing in ChromaDB...")
    embed_and_store(chunks)

    # ── Summary ─────────────────────────────────────────────
    stats = get_collection_stats()
    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE!")
    print(f"   Articles scraped:  {len(articles)}")
    print(f"   Chunks created:    {len(chunks)}")
    print(f"   Vectors stored:    {stats['document_count']}")
    print(f"   ChromaDB location: {stats['persist_dir']}")
    print("=" * 60)
    print("\n💡 You can now query the RAG system!")
    print("   Run: python -c \"from src.pipeline.rag_pipeline import query_rag; print(query_rag('What are the top Indian unicorns?')['answer'])\"")


if __name__ == "__main__":
    main()
