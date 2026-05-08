"""
=============================================================
TEST_RETRIEVAL.PY — Test Qdrant Retrieval (No LLM Needed)
=============================================================

USAGE:
    python scripts/test_retrieval.py

WHAT IT DOES:
    Tests JUST the retrieval pipeline (embedding query + Qdrant search).
    This does NOT call the LLM for generation, so it won't hit
    the generation API quota.

    Use this to verify that:
    1. Your embeddings were stored correctly in Qdrant
    2. Similarity search returns relevant results
    3. Scores are reasonable (>0.6 for relevant queries)

WHY SEPARATE FROM FULL PIPELINE?
    The Gemini free tier has TWO separate quotas:
    - Embedding API quota (for embed_query / embed_documents)
    - Generation API quota (for chat/generation)
    
    When the generation quota is exhausted, you can still test
    retrieval separately. This is also good practice —
    test components in ISOLATION before testing the full pipeline.
=============================================================
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.basic_retriever import retrieve

# ── Test Queries ────────────────────────────────────────────
# These queries test different retrieval capabilities:
# 1. Factual recall: finding specific facts
# 2. Conceptual: understanding a concept
# 3. Multi-source: requiring information from multiple articles
TEST_QUERIES = [
    "Who founded Zerodha?",
    "What is UPI and how does it work?",
    "Tell me about Flipkart's acquisition",
    "Which Indian startups are bootstrapped?",
    "What is BYJU'S valuation?",
]

def main():
    print("=" * 60)
    print("🔍 RETRIEVAL TEST (No LLM, No Generation Quota)")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"{'─' * 60}")

        try:
            chunks = retrieve(query, top_k=5)
            for i, chunk in enumerate(chunks, 1):
                title = chunk["metadata"]["title"]
                score = chunk["score"]
                preview = chunk["text"][:120].replace("\n", " ")
                print(f"  [{i}] Score: {score:.3f} | {title}")
                print(f"      → {preview}...")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'=' * 60}")
    print("✅ Retrieval test complete!")
    print("   If scores > 0.6 for relevant queries, retrieval is working well.")
    print("   Run the full pipeline test when API quota resets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
