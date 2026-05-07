"""
=============================================================
EMBEDDER.PY — Embedding + Vector Storage
=============================================================

WHAT THIS DOES:
    Takes chunked text, converts it to vector embeddings, and
    stores those vectors in ChromaDB for similarity search.

THE BIG PICTURE — WHY EMBEDDINGS?
    Traditional search (SQL LIKE, regex) does EXACT matching:
    - Query: "startup funding" ✓ matches "startup funding"
    - Query: "startup funding" ✗ does NOT match "raised capital"

    But "startup funding" and "raised capital" MEAN THE SAME THING.

    Embeddings solve this by converting text → numbers (vectors)
    that capture MEANING, not just words:
    - "startup funding" → [0.82, 0.15, -0.43, ...]  (768 numbers)
    - "raised capital"  → [0.80, 0.17, -0.41, ...]  (similar numbers!)
    - "cat food"        → [-0.12, 0.91, 0.33, ...]  (very different!)

    When we search, we find vectors that are CLOSE to the query vector.
    Close vectors = similar meaning. This is SEMANTIC search.

HOW COSINE SIMILARITY WORKS:
    - Two vectors point in a "direction" in 768-dimensional space
    - Cosine similarity measures the ANGLE between them
    - cos(0°) = 1.0 → identical meaning
    - cos(90°) = 0.0 → unrelated
    - We don't care about vector LENGTH, only direction
    - This is why it works better than Euclidean distance for text

WHY VECTOR DB INSTEAD OF SQL?
    - SQL: stores rows and columns, searches with WHERE clauses
    - Vector DB: stores vectors, searches with "find nearest neighbors"
    - SQL can't do "find the 10 most semantically similar documents"
    - Vector DBs use algorithms like HNSW (Hierarchical Navigable
      Small World) to make nearest-neighbor search fast even with
      millions of vectors

WHY CHROMADB?
    - Runs locally in your Python process (no server needed)
    - Persists to disk (survives restarts)
    - Free and open source
    - Good enough for up to ~1M documents
    - In production, companies use: Pinecone (managed), Qdrant (self-hosted),
      Weaviate, Milvus, pgvector (Postgres extension)

INTERVIEW QUESTION:
    "Why not just use a regular database?"
    → "Regular databases can't do semantic similarity search. Vector
       databases use embedding vectors and approximate nearest neighbor
       algorithms to find semantically similar documents in O(log n) time."
=============================================================
"""

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
)


def get_embedding_function():
    """
    Create the embedding function using Google's Gemini.

    WHAT IS AN EMBEDDING MODEL?
        - A neural network trained to convert text → fixed-size vectors
        - Google's gemini-embedding-001 produces 768-dimensional vectors
        - It was trained on billions of text pairs to learn that
          semantically similar texts should have similar vectors

    WHY GEMINI EMBEDDINGS (not Sentence Transformers)?
        - Free API, no GPU needed for embedding
        - High quality (trained by Google on massive data)
        - 768 dimensions is a good balance (captures nuance without
          being too large to store/search)
        - Alternative: sentence-transformers/all-MiniLM-L6-v2 (local,
          384-dim, slightly lower quality but no API needed)

    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_chroma_client():
    """
    Create a persistent ChromaDB client.

    PERSISTENCE:
        - PersistentClient saves vectors to disk at CHROMA_PERSIST_DIR
        - Without persistence, all your embeddings would be lost when
          the Python process exits
        - You'd have to re-embed everything (slow + costs API calls)

    Returns:
        chromadb.PersistentClient
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def embed_and_store(chunks: list[dict], batch_size: int = 50) -> None:
    """
    Embed all chunks and store them in ChromaDB.

    THE EMBEDDING PIPELINE:
        1. Take each chunk's text
        2. Send to Google's embedding API → get back a 768-dim vector
        3. Store the vector + original text + metadata in ChromaDB

    WHY BATCH?
        - Sending 1 chunk at a time = 500 API calls (slow, rate-limited)
        - Sending 50 at a time = 10 API calls (fast, efficient)
        - Google's API supports batch embedding
        - batch_size=50 balances speed vs memory usage

    WHAT GETS STORED IN CHROMADB:
        - ids: unique identifier for each chunk (for deduplication)
        - documents: the original text (returned with search results)
        - metadatas: article title, URL, etc. (for citations)
        - embeddings: the 768-dim vectors (for similarity search)

    Args:
        chunks: List of chunk dicts from chunker.py
        batch_size: Number of chunks to embed in each API call
    """
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    # Get or create the collection
    # If it already exists (from a previous run), we add to it
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    print(f"🔢 Embedding {len(chunks)} chunks (batch_size={batch_size})...")

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["id"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]

        # Call Google's embedding API
        # This converts each text → 768-dimensional vector
        embeddings = embedding_fn.embed_documents(texts)

        # Upsert into ChromaDB
        # "upsert" = update if exists, insert if new
        # This makes the script idempotent (safe to run multiple times)
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")
    print(f"   Total documents in collection: {collection.count()}")


def get_collection_stats() -> dict:
    """
    Get statistics about the ChromaDB collection.

    Useful for debugging and verification.

    Returns:
        dict with collection name and document count
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        return {
            "collection_name": CHROMA_COLLECTION_NAME,
            "document_count": collection.count(),
            "persist_dir": CHROMA_PERSIST_DIR,
        }
    except Exception:
        return {
            "collection_name": CHROMA_COLLECTION_NAME,
            "document_count": 0,
            "persist_dir": CHROMA_PERSIST_DIR,
            "status": "Collection not found — run ingestion first",
        }
