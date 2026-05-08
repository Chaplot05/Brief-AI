"""
Test Stage 4: HyDE (Hypothetical Document Embeddings).

Compares three retrieval strategies:
1. Basic (Stage 1-2): embed question → search
2. Multi-Query (Stage 3): embed 4 question variants → search
3. HyDE (Stage 4): embed hypothetical answer + question → search

Shows which chunks each method finds, highlighting unique discoveries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.basic_retriever import retrieve
from src.retrieval.hyde_retriever import hyde_retrieve, generate_hypothetical_answer

QUERY = "What are the challenges faced by Indian edtech companies?"

print("=" * 60)
print("STAGE 4: HyDE RETRIEVAL TEST")
print("=" * 60)

# ── Test 1: See the hypothetical answer ──────────────
print("\n--- Step 1: Generate hypothetical answer ---")
try:
    hypothesis = generate_hypothetical_answer(QUERY)
    if hypothesis:
        print(f"  Question:   {QUERY}")
        print(f"  Hypothesis: {hypothesis}")
    else:
        print("  Could not generate hypothesis (LLM quota may be exhausted)")
except Exception as e:
    print(f"  Error: {e}")
    hypothesis = None

# ── Test 2: Basic retrieval ──────────────────────────
print("\n--- Step 2: Basic retrieval (Stage 1-2) ---")
basic_chunks = retrieve(QUERY, top_k=5)
basic_ids = set()
for i, c in enumerate(basic_chunks, 1):
    basic_ids.add(c["id"])
    company = c["metadata"].get("company_name", "?")
    print(f"  [{i}] Score: {c['score']:.3f} | {company}")
    print(f"      {c['text'][:80]}...")

# ── Test 3: HyDE retrieval ───────────────────────────
print("\n--- Step 3: HyDE retrieval (Stage 4) ---")
try:
    hyde_chunks = hyde_retrieve(QUERY, top_k=5, verbose=True)
    print("\n  Final results:")
    hyde_ids = set()
    for i, c in enumerate(hyde_chunks, 1):
        hyde_ids.add(c["id"])
        company = c["metadata"].get("company_name", "?")
        is_new = " ★NEW" if c["id"] not in basic_ids else ""
        print(f"  [{i}] Score: {c['score']:.3f} | {company}{is_new}")
        print(f"      {c['text'][:80]}...")

    # Show improvement
    new_chunks = hyde_ids - basic_ids
    print(f"\n  HyDE found {len(new_chunks)} chunks that basic retrieval missed!")
except Exception as e:
    print(f"  HyDE failed: {e}")
    print("  This is expected if LLM quota is exhausted.")
    print("  The code will still search with original query embedding.")

print("\n" + "=" * 60)
print("Stage 4 test complete!")
print("=" * 60)
