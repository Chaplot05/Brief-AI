"""
=============================================================
TEST_CHUNK_SIZES.PY — Compare chunk sizes to find the best one
=============================================================

WHAT THIS DOES:
    Tests chunk sizes 200, 400, 600 on the same data and shows
    how many chunks each produces plus average chunk quality.

WHY TEST?
    Saying "I used 400 because the internet said so" is weak.
    Saying "I tested 200, 400, 600 and found 400 optimal" is strong.
    This script gives you DATA to back up your decision.
=============================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.scraper import load_scraped_articles
from src.ingestion.chunker import chunk_text, count_tokens


def test_chunk_sizes():
    """Compare different chunk sizes on the same articles."""
    articles = load_scraped_articles()
    if not articles:
        print("No articles found. Run seed_data.py or pdf_loader.py first.")
        return

    # Combine all article text
    all_texts = [a["text"] for a in articles if a.get("text")]

    sizes_to_test = [200, 300, 400, 500, 600, 800]

    print("=" * 70)
    print("CHUNK SIZE COMPARISON")
    print("=" * 70)
    print(f"Testing on {len(all_texts)} articles\n")
    print(f"{'Chunk Size':>10} | {'Overlap':>7} | {'# Chunks':>8} | {'Avg Tokens':>10} | {'Min':>5} | {'Max':>5}")
    print("-" * 70)

    results = {}

    for size in sizes_to_test:
        overlap = size // 8  # Overlap = 1/8th of chunk size (standard ratio)
        all_chunks = []

        for text in all_texts:
            chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            continue

        token_counts = [count_tokens(c) for c in all_chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

        results[size] = {
            "num_chunks": len(all_chunks),
            "avg_tokens": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "overlap": overlap,
        }

        print(f"{size:>10} | {overlap:>7} | {len(all_chunks):>8} | {avg_tokens:>10.1f} | {min_tokens:>5} | {max_tokens:>5}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
WHAT THE NUMBERS MEAN:

  Fewer chunks (large chunk size):
    + Less storage, fewer API calls to embed
    + Each chunk has more context
    - Retrieval is LESS precise (chunks contain mixed topics)
    - More noise sent to the LLM

  More chunks (small chunk size):
    + Retrieval is MORE precise (chunks are focused)
    + Less noise in LLM context
    - More storage, more API calls
    - Risk of cutting sentences mid-thought

RECOMMENDATION:
  400 tokens with 50 overlap is the sweet spot for most RAG systems.
  But YOUR data might be different — test with your actual queries!
""")

    # Show example chunks at different sizes
    sample_text = all_texts[0]  # First article
    print("=" * 70)
    print(f"EXAMPLE: First article chunked at different sizes")
    print(f"Article: {articles[0].get('title', 'Unknown')[:50]}")
    print("=" * 70)

    for size in [200, 400, 600]:
        overlap = size // 8
        chunks = chunk_text(sample_text, chunk_size=size, chunk_overlap=overlap)
        print(f"\n--- CHUNK SIZE = {size} (produced {len(chunks)} chunks) ---")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks only
            tokens = count_tokens(chunk)
            preview = chunk[:150].replace("\n", " ")
            print(f"  Chunk {i+1} ({tokens} tokens): {preview}...")


if __name__ == "__main__":
    test_chunk_sizes()
