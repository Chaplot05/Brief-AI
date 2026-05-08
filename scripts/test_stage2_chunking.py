"""Quick test of Stage 2 chunking — compare with Stage 1 results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.scraper import load_scraped_articles
from src.ingestion.chunker import chunk_articles

articles = load_scraped_articles()
chunks = chunk_articles(articles)

print(f"\nTotal chunks: {len(chunks)}")
print(f"(Stage 1 had: 166 chunks)")

# Show sample metadata
sample = chunks[0]
print(f"\nSample metadata keys: {list(sample['metadata'].keys())}")
print(f"  source_type:  {sample['metadata'].get('source_type', 'N/A')}")
print(f"  company_name: {sample['metadata'].get('company_name', 'N/A')}")
print(f"  title:        {sample['metadata'].get('title', 'N/A')}")

# Show first 300 chars of first chunk
print(f"\nFirst chunk preview (300 chars):")
print(sample["text"][:300])

# Count by source type
from collections import Counter
source_counts = Counter(c["metadata"]["source_type"] for c in chunks)
print(f"\nChunks by source type: {dict(source_counts)}")

# Count by company
company_counts = Counter(c["metadata"]["company_name"] for c in chunks)
print(f"\nChunks by company (top 10):")
for company, count in company_counts.most_common(10):
    print(f"  {company}: {count} chunks")
