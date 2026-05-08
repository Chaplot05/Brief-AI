"""
=============================================================
BASIC_RETRIEVER.PY — Vector Similarity Search (Qdrant)
=============================================================

WHAT THIS DOES:
    Takes a user query, embeds it, and finds the most similar
    chunks in Qdrant using cosine similarity.

THIS IS THE "R" IN RAG:
    RAG = Retrieval-Augmented Generation
    - Retrieval: find relevant documents (THIS FILE)
    - Augmented: add those documents to the LLM's context
    - Generation: LLM generates an answer using the context

HOW RETRIEVAL WORKS:
    1. User asks: "Who founded Zerodha?"
    2. We embed the query → get a 768-dim vector
    3. Qdrant finds the 10 chunks whose vectors are closest
       to the query vector (using cosine similarity)
    4. We return those chunks as "context" for the LLM

CHROMADB vs QDRANT — RETRIEVAL DIFFERENCES:
    ChromaDB:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )
        # Returns parallel lists: results["documents"][0], results["distances"][0]
        # Distances need conversion: similarity = 1 - distance

    Qdrant:
        results = client.query_points(
            collection_name="...",
            query=query_vec,
            limit=10
        )
        # Returns ScoredPoint objects with .score (already similarity!)
        # Access data via .payload dict — much cleaner

    KEY DIFFERENCE: ChromaDB returns "distance" (lower = more similar),
    but Qdrant returns "score" (higher = more similar, already in 0-1 range
    for cosine). No manual conversion needed!

WHY THIS IS BETTER THAN KEYWORD SEARCH:
    Query: "Who founded Zerodha?"
    - Keyword search: only finds chunks containing "founded" AND "Zerodha"
    - Vector search: finds chunks about "Nithin Kamath started Zerodha",
      "the creation of Zerodha", "Zerodha's origin story"
    - Vector search understands MEANING, not just words

TOP-K: WHY 10?
    - We want enough candidates to ensure we don't miss the answer
    - But not so many that we overwhelm the LLM with noise
    - In Stage 5, we'll retrieve 20, rerank with BGE, keep 5
    - For basic RAG, top-10 is a good starting point

INTERVIEW QUESTION:
    "What happens if the relevant document isn't in the top-k?"
    → "This is called a 'retrieval failure'. Solutions include:
       increasing k, using multi-query (Stage 3), HyDE (Stage 4),
       or hybrid search (vector + keyword). We implement all of these."
=============================================================
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    source_type: str | None = None,
    company_name: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    STAGE 2 UPGRADE: FILTERED SEARCH
        Now supports optional metadata filters:
        - source_type: "wikipedia", "yourstory", etc.
        - company_name: "Zerodha", "Flipkart", etc.

        This combines VECTOR SEARCH (semantic similarity) with
        METADATA FILTERING (exact constraints). Example:
        - "Find chunks about funding" + source_type="wikipedia"
        - Returns only Wikipedia chunks that are semantically
          similar to "funding"

        WHY IS THIS POWERFUL?
        Without filters: query about Zerodha might return chunks
        about Flipkart or Paytm that mention similar concepts.
        With filters: constrain to only Zerodha articles → precise.

        This is called HYBRID SEARCH (vector + filter) and it's
        a key advantage of Qdrant over ChromaDB.

    THE RETRIEVAL PIPELINE:
        1. Embed the query using the SAME embedding model used for indexing
           (CRITICAL: query and documents MUST use the same model)
        2. Build optional Qdrant filter from metadata params
        3. Qdrant searches ONLY matching documents for similarity
        4. Returns the top-k most similar chunks as ScoredPoint objects

    WHAT IS HNSW?
        Hierarchical Navigable Small World — the algorithm Qdrant uses
        for approximate nearest neighbor search. Without it, finding
        the closest vector requires comparing against ALL vectors (O(n)).
        HNSW builds a graph structure that finds approximate nearest
        neighbors in O(log n) time. The tradeoff: ~99% accuracy vs
        100x faster search. Every production vector DB uses this.

    INTERVIEW QUESTION:
        "How do you handle filtered vector search?"
        → "Qdrant supports payload-based filtering alongside HNSW
           vector search. I can pass a Filter with FieldConditions
           to restrict search to specific source types or companies.
           This runs at the vector DB level, not in post-processing,
           so it's efficient even with millions of documents."

    Args:
        query: The user's question
        top_k: Number of chunks to retrieve
        source_type: Optional filter — only search this source type
        company_name: Optional filter — only search this company

    Returns:
        List of dicts, each containing:
            - text: chunk text
            - metadata: article info (title, URL, source_type, etc.)
            - score: similarity score (0-1, higher = more similar)
            - id: chunk identifier
    """
    # Initialize embedding function (SAME model as used in indexing)
    embedding_fn = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # Embed the query
    query_embedding = embedding_fn.embed_query(query)

    # Connect to Qdrant (local mode — same path as indexing)
    client = QdrantClient(path=QDRANT_PATH)

    # ── Build Qdrant filter (Stage 2) ──────────────────────
    # Qdrant filters work at the DATABASE level, not in post-processing.
    # This means:
    #   1. Qdrant first narrows down to matching documents
    #   2. THEN does vector search only within those documents
    #   3. Much faster than retrieving everything and filtering after
    #
    # FieldCondition: matches exact values in payload fields
    # Filter(must=[...]): ALL conditions must be true (AND logic)
    # Filter(should=[...]): ANY condition can be true (OR logic)
    query_filter = None
    conditions = []
    if source_type:
        conditions.append(
            FieldCondition(key="source_type", match=MatchValue(value=source_type))
        )
    if company_name:
        conditions.append(
            FieldCondition(key="company_name", match=MatchValue(value=company_name))
        )
    if conditions:
        query_filter = Filter(must=conditions)

    # Query Qdrant for the top-k most similar chunks
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,  # None = no filter (search all)
        with_payload=True,
    )

    # Format results into a clean list of dicts
    # Dynamically extract metadata from payload (future-proof)
    INTERNAL_FIELDS = {"text", "chunk_id"}  # Not metadata
    retrieved_chunks = []
    for point in results.points:
        # Build metadata dict from all non-internal payload fields
        metadata = {
            k: v for k, v in point.payload.items()
            if k not in INTERNAL_FIELDS
        }

        retrieved_chunks.append({
            "id": point.payload.get("chunk_id", str(point.id)),
            "text": point.payload["text"],
            "metadata": metadata,
            "score": point.score,
        })

    return retrieved_chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    WHY FORMAT MATTERS:
        - The LLM needs clear structure to understand which source
          says what
        - Numbering chunks helps the LLM cite specific sources
        - Including source titles helps the LLM attribute information

    STAGE 2 UPDATE:
        Now shows company_name (if available) for better source display.

    Args:
        chunks: List of retrieved chunk dicts

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        # Use company_name if available (Stage 2), otherwise fall back to title
        source = meta.get("company_name", meta.get("title", "Unknown Source"))
        url = meta.get("url", "")
        score = chunk.get("score", 0)
        source_type = meta.get("source_type", "")

        header = f"[Source {i}] (Relevance: {score:.2f}) — {source}"
        if source_type:
            header += f" [{source_type}]"

        context_parts.append(
            f"{header}\n"
            f"URL: {url}\n"
            f"{chunk['text']}\n"
        )

    return "\n---\n".join(context_parts)
