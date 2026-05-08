"""
=============================================================
EMBEDDER.PY — Embedding + Qdrant Vector Storage
=============================================================

WHAT THIS DOES:
    Takes chunked text, converts it to vector embeddings using
    Google's Gemini API, and stores those vectors in Qdrant
    for similarity search.

MIGRATION: CHROMADB → QDRANT
    We migrated from ChromaDB to Qdrant. Here's what changed:

    ChromaDB way:
        collection.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)
        - ChromaDB handles documents, metadata, and embeddings as separate lists
        - Auto-infers vector dimensions

    Qdrant way:
        client.upsert(collection_name, points=[PointStruct(id, vector, payload)])
        - Each point is a self-contained object with id + vector + payload
        - Payload = your metadata + original text (you choose what to store)
        - You must declare vector dimensions upfront when creating collection

    WHICH IS BETTER?
        Qdrant's approach is more explicit and production-friendly:
        - You know EXACTLY what's stored in each point
        - Payload can have complex nested structures
        - Payload fields can be INDEXED for filtered search
        - No ambiguity about what "documents" vs "metadatas" contain

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

WHY QDRANT INSTEAD OF CHROMADB?
    - ChromaDB: embedded SQLite, fine for <1M docs, no server mode
    - Qdrant: production-grade, client-server architecture, Docker-ready
    - Qdrant supports payload indexing for filtered search
    - In local mode, Qdrant saves to disk just like ChromaDB did
    - When we Docker-ize (Stage 9), Qdrant runs as its own container

INTERVIEW QUESTION:
    "Why not just use a regular database?"
    → "Regular databases can't do semantic similarity search. Vector
       databases use embedding vectors and approximate nearest neighbor
       algorithms to find semantically similar documents in O(log n) time."

    "Why Qdrant over ChromaDB?"
    → "ChromaDB is great for prototyping but it's embedded (SQLite-based).
       Qdrant has a client-server architecture that maps naturally to
       microservices and Docker. It also supports payload indexing for
       filtered vector search, which ChromaDB doesn't do well."
=============================================================
"""

import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
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


def get_qdrant_client() -> QdrantClient:
    """
    Create a persistent Qdrant client in LOCAL mode.

    TWO MODES OF QDRANT:
        1. LOCAL MODE (what we use now):
           QdrantClient(path="./qdrant_db")
           - Saves vectors to disk at QDRANT_PATH
           - No server needed — runs in your Python process
           - Perfect for development and testing
           - Data persists across restarts

        2. SERVER MODE (Stage 9 — Docker):
           QdrantClient(url="http://qdrant:6333")
           - Connects to a Qdrant server running in Docker
           - Supports multi-client access, clustering, snapshots
           - We'll switch to this when we containerize

    WHY LOCAL MODE FOR NOW?
        - Zero setup — no Docker, no server, just pip install
        - Same API as server mode — zero code changes when we switch
        - Fast development iteration
        - Your vectors survive Python restarts (saved to disk)

    Returns:
        QdrantClient instance
    """
    return QdrantClient(path=QDRANT_PATH)


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """
    Convert a string chunk ID to a deterministic UUID.

    WHY UUIDs?
        Qdrant supports two types of point IDs:
        1. Unsigned integers (0, 1, 2, ...)
        2. UUID strings ("550e8400-e29b-41d4-a716-446655440000")

        Our chunk IDs look like "wiki_620ea58ad4_chunk_0" — these
        are neither integers nor UUIDs. So we need to convert them.

    WHY uuid5 (not uuid4)?
        - uuid4() generates a RANDOM UUID every time — not deterministic
        - uuid5() generates a UUID from a NAME — same input = same UUID
        - This makes our embedding IDEMPOTENT:
          Running ingest.py twice produces the same UUIDs, so Qdrant
          does an "upsert" (update) instead of creating duplicates

    WHAT IS IDEMPOTENT?
        An operation that produces the same result no matter how many
        times you run it. This is CRITICAL for data pipelines:
        - If ingest.py crashes halfway, you re-run it safely
        - If you add new articles, existing embeddings aren't duplicated
        - Every production data pipeline MUST be idempotent

    Args:
        chunk_id: String ID like "wiki_620ea58ad4_chunk_0"

    Returns:
        UUID string like "550e8400-e29b-41d4-a716-446655440000"
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def embed_and_store(chunks: list[dict], batch_size: int = 20) -> None:
    """
    Embed all chunks and store them in Qdrant.

    THE EMBEDDING PIPELINE:
        1. Take each chunk's text
        2. Send to Google's embedding API → get back a 768-dim vector
        3. Package into a PointStruct (id + vector + payload)
        4. Upsert into Qdrant collection

    WHY BATCH?
        - Sending 1 chunk at a time = 166 API calls (slow, rate-limited)
        - Sending 50 at a time = 4 API calls (fast, efficient)
        - Google's API supports batch embedding
        - batch_size=50 balances speed vs memory usage

    QDRANT'S DATA MODEL:
        Each "point" in Qdrant has three parts:
        1. ID: unique identifier (UUID in our case)
        2. VECTOR: the 768-dim embedding (for similarity search)
        3. PAYLOAD: any JSON data you want to store alongside
           - We store the original text (so we can return it in results)
           - We store metadata (title, URL, chunk_index — for citations)
           - Payload fields can be INDEXED for filtered search

    CHROMADB vs QDRANT — HOW DATA IS STORED:
        ChromaDB: collection.upsert(
            ids=["chunk_0", "chunk_1"],           # list of IDs
            documents=["text0", "text1"],          # list of texts
            metadatas=[{...}, {...}],              # list of metadata dicts
            embeddings=[[0.1, ...], [0.2, ...]]    # list of vectors
        )
        → Four parallel lists. Fragile — easy to misalign them.

        Qdrant: client.upsert(collection_name, points=[
            PointStruct(
                id="uuid-1",                      # self-contained
                vector=[0.1, ...],                 # vector
                payload={"text": "...", ...}       # everything else
            ),
        ])
        → Each point is one object. Impossible to misalign.

    Args:
        chunks: List of chunk dicts from chunker.py
        batch_size: Number of chunks to embed in each API call
    """
    client = get_qdrant_client()
    embedding_fn = get_embedding_function()

    # ── Create collection if it doesn't exist ──────────────────
    # Unlike ChromaDB's "get_or_create_collection()", Qdrant requires
    # explicit collection creation with vector configuration.
    #
    # VectorParams specifies:
    #   - size: dimension of vectors (768 for Gemini)
    #   - distance: similarity metric (COSINE for text search)
    #
    # WHY COSINE (not Euclidean or Dot Product)?
    #   - Cosine: measures angle between vectors (direction = meaning)
    #   - Euclidean: measures distance (affected by vector magnitude)
    #   - Dot Product: faster but affected by magnitude
    #   - For text embeddings, COSINE is standard because we care
    #     about semantic direction, not vector length
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        print(f"📦 Created Qdrant collection: {QDRANT_COLLECTION_NAME}")
    else:
        print(f"📦 Using existing collection: {QDRANT_COLLECTION_NAME}")

    print(f"🔢 Embedding {len(chunks)} chunks (batch_size={batch_size})...")

    total_embedded = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
        batch = chunks[i : i + batch_size]

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Call Google's embedding API with retry logic
        # This converts each text → 3072-dimensional vector
        #
        # RATE LIMITING — WHY THIS MATTERS:
        #   Google's free tier has rate limits (requests per minute).
        #   If we blast 166 chunks at the API too fast, we get 429 errors.
        #   The fix: smaller batches + delays + retry with backoff.
        #
        # EXPONENTIAL BACKOFF:
        #   - 1st retry: wait 30s
        #   - 2nd retry: wait 60s
        #   - 3rd retry: wait 90s
        #   Every production system that calls external APIs uses this.
        #   AWS, Google, Stripe — they all recommend this pattern.
        embeddings = None
        for attempt in range(4):  # 1 try + 3 retries
            try:
                embeddings = embedding_fn.embed_documents(texts)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 30 * (attempt + 1)
                    print(f"\n⏳ Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/3)")
                    time.sleep(wait_time)
                else:
                    raise  # Re-raise non-rate-limit errors

        if embeddings is None:
            print(f"\n❌ Failed to embed batch {i // batch_size + 1} after 3 retries. Skipping.")
            continue

        # Build Qdrant PointStruct objects
        # Each point = one chunk with its vector and all metadata
        points = []
        for chunk, embedding in zip(batch, embeddings):
            # Build payload from chunk metadata
            # We flatten ALL metadata fields into the payload so they're
            # searchable and filterable in Qdrant.
            #
            # Stage 2 NEW FIELDS:
            #   - source_type: "wikipedia", "yourstory", etc. (for filtered search)
            #   - company_name: "Zerodha", "Flipkart", etc. (for filtered search)
            payload = {
                "text": chunk["text"],
                "chunk_id": chunk["id"],
            }
            # Merge all metadata fields into payload
            # This way, any new metadata added in future stages
            # automatically gets stored — no need to edit this code
            payload.update(chunk["metadata"])

            points.append(
                PointStruct(
                    id=_chunk_id_to_uuid(chunk["id"]),
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert into Qdrant
        # "upsert" = update if ID exists, insert if new
        # This makes the script idempotent (safe to run multiple times)
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )
        total_embedded += len(batch)

        # Small delay between batches to avoid hitting rate limits
        # This is "rate limiting on the client side" — a best practice
        # when calling any external API with quotas
        if i + batch_size < len(chunks):
            time.sleep(2)

    # Get final count
    collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
    total_points = collection_info.points_count

    print(f"✅ Stored {total_embedded} chunks in Qdrant")
    print(f"   Collection: {QDRANT_COLLECTION_NAME}")
    print(f"   Total points in collection: {total_points}")


def get_collection_stats() -> dict:
    """
    Get statistics about the Qdrant collection.

    Useful for debugging and verification.

    WHAT QDRANT TELLS US:
        - points_count: total number of vectors stored
        - vectors_count: total vectors (same as points for single-vector)
        - status: "green" = healthy, "yellow" = optimizing
        - optimizer_status: whether background optimization is running

    Returns:
        dict with collection name and document count
    """
    client = get_qdrant_client()
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
        return {
            "collection_name": QDRANT_COLLECTION_NAME,
            "document_count": collection_info.points_count,
            "persist_dir": QDRANT_PATH,
            "status": str(collection_info.status),
        }
    except Exception:
        return {
            "collection_name": QDRANT_COLLECTION_NAME,
            "document_count": 0,
            "persist_dir": QDRANT_PATH,
            "status": "Collection not found — run ingestion first",
        }
