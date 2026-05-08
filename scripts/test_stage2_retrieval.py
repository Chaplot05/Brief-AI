"""Test Stage 2 features: improved chunking + filtered search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.basic_retriever import retrieve

print("=" * 60)
print("STAGE 2 RETRIEVAL TEST")
print("=" * 60)

# Test 1: Normal search (same as Stage 1)
print("\n--- Test 1: Normal search ---")
chunks = retrieve("Who founded Zerodha?", top_k=3)
for i, c in enumerate(chunks, 1):
    print(f"  [{i}] Score: {c['score']:.3f} | {c['metadata'].get('company_name', '?')} [{c['metadata'].get('source_type', '?')}]")
    print(f"      {c['text'][:100]}...")

# Test 2: Filtered by company_name (NEW in Stage 2)
print("\n--- Test 2: Filtered search (company_name='Wikipedia') ---")
chunks = retrieve("funding and valuation", top_k=3, company_name="Wikipedia")
for i, c in enumerate(chunks, 1):
    print(f"  [{i}] Score: {c['score']:.3f} | {c['metadata'].get('company_name', '?')}")

# Test 3: Filtered by company (specific)
print("\n--- Test 3: Filtered search (company_name='Unified Payments Interface') ---")
chunks = retrieve("how does payment work?", top_k=3, company_name="Unified Payments Interface")
for i, c in enumerate(chunks, 1):
    print(f"  [{i}] Score: {c['score']:.3f} | {c['metadata'].get('company_name', '?')}")
    print(f"      {c['text'][:100]}...")

print("\n" + "=" * 60)
print("Stage 2 retrieval test complete!")
print("=" * 60)
