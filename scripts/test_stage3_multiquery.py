"""
Test Stage 3: Multi-Query Retrieval.

Tests whether generating multiple query variants improves retrieval
by finding more diverse, relevant chunks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.multi_query_retriever import multi_query_retrieve, generate_query_variants
from src.retrieval.basic_retriever import retrieve

QUERY = "How much money has Zerodha raised?"

print("=" * 60)
print("STAGE 3: MULTI-QUERY RETRIEVAL TEST")
print("=" * 60)

# ── Test 1: Generate query variants ──────────────────
print("\n--- Step 1: Generating query variants ---")
try:
    variants = generate_query_variants(QUERY, num_queries=3)
    if variants:
        print(f"  Original: {QUERY}")
        for i, v in enumerate(variants, 1):
            print(f"  Variant {i}: {v}")
    else:
        print("  No variants generated (LLM quota may be exhausted)")
        print("  Falling back to comparison test with basic retrieval only")
except Exception as e:
    print(f"  Error: {e}")
    variants = []

# ── Test 2: Compare basic vs multi-query ─────────────
print("\n--- Step 2: Basic retrieval (Stage 1-2) ---")
basic_chunks = retrieve(QUERY, top_k=5)
basic_ids = set()
for i, c in enumerate(basic_chunks, 1):
    basic_ids.add(c["id"])
    company = c["metadata"].get("company_name", "?")
    print(f"  [{i}] Score: {c['score']:.3f} | {company}")
    print(f"      {c['text'][:80]}...")

if variants:
    print("\n--- Step 3: Multi-query retrieval (Stage 3) ---")
    try:
        multi_chunks = multi_query_retrieve(QUERY, num_queries=3, top_k=5, verbose=True)
        multi_ids = set()
        print("\n  Final results:")
        for i, c in enumerate(multi_chunks, 1):
            multi_ids.add(c["id"])
            company = c["metadata"].get("company_name", "?")
            is_new = "NEW" if c["id"] not in basic_ids else "   "
            print(f"  [{i}] Score: {c['score']:.3f} | {company} {is_new}")
            print(f"      {c['text'][:80]}...")

        # Show improvement
        new_chunks = multi_ids - basic_ids
        print(f"\n  Improvement: {len(new_chunks)} new chunks found by multi-query!")
        print(f"  Basic found: {len(basic_ids)} unique | Multi found: {len(multi_ids)} unique")
    except Exception as e:
        print(f"  Multi-query failed: {e}")
else:
    print("\n--- Step 3: Skipped (LLM unavailable for variant generation) ---")
    print("  Multi-query retrieval requires LLM access to generate variants.")
    print("  The code is ready — test when API quota resets.")

print("\n" + "=" * 60)
print("Stage 3 test complete!")
print("=" * 60)
