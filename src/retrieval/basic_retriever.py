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
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    THE RETRIEVAL PIPELINE:
        1. Embed the query using the SAME embedding model used for indexing
           (CRITICAL: query and documents MUST use the same model,
            otherwise the vector spaces don't align and similarity
            scores become meaningless)
        2. Send query vector to Qdrant
        3. Qdrant computes cosine similarity between query vector
           and ALL stored vectors using its HNSW index
        4. Returns the top-k most similar chunks as ScoredPoint objects

    WHAT IS HNSW?
        Hierarchical Navigable Small World — the algorithm Qdrant uses
        for approximate nearest neighbor search. Without it, finding
        the closest vector requires comparing against ALL vectors (O(n)).
        HNSW builds a graph structure that finds approximate nearest
        neighbors in O(log n) time. The tradeoff: ~99% accuracy vs
        100x faster search. Every production vector DB uses some form
        of ANN (Approximate Nearest Neighbor) algorithm.

    QDRANT'S query_points() vs search():
        - query_points(): newer API, returns QueryResponse with .points
        - search(): older API, returns list of ScoredPoint directly
        We use query_points() as it's the recommended modern API.

    Args:
        query: The user's question
        top_k: Number of chunks to retrieve

    Returns:
        List of dicts, each containing:
            - text: chunk text
            - metadata: article info (title, URL, etc.)
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

    # Query Qdrant for the top-k most similar chunks
    # query_points() is Qdrant's modern search API
    #
    # WHAT HAPPENS UNDER THE HOOD:
    #   1. Qdrant receives the query vector
    #   2. It traverses the HNSW graph to find approximate nearest neighbors
    #   3. For each candidate, it computes exact cosine similarity
    #   4. Returns the top-k results sorted by score (highest first)
    #
    # SCORE INTERPRETATION (Cosine):
    #   1.0 = identical vectors (same meaning)
    #   0.7+ = highly relevant
    #   0.5-0.7 = somewhat relevant
    #   <0.5 = probably not relevant
    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,  # Include the stored text + metadata
    )

    # Format results into a clean list of dicts
    # Qdrant returns ScoredPoint objects — much cleaner than ChromaDB's
    # parallel lists. Each point has: .id, .score, .payload, .version
    retrieved_chunks = []
    for point in results.points:
        retrieved_chunks.append({
            "id": point.payload.get("chunk_id", str(point.id)),
            "text": point.payload["text"],
            "metadata": {
                "article_id": point.payload.get("article_id", ""),
                "title": point.payload.get("title", "Unknown"),
                "url": point.payload.get("url", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "total_chunks": point.payload.get("total_chunks", 0),
                "token_count": point.payload.get("token_count", 0),
            },
            # Qdrant's .score is ALREADY a similarity score (0-1 for cosine)
            # No conversion needed! ChromaDB returned "distance" which
            # we had to convert via (1 - distance). Qdrant is cleaner.
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

    Args:
        chunks: List of retrieved chunk dicts

    Returns:
        Formatted context string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("title", "Unknown Source")
        url = chunk["metadata"].get("url", "")
        score = chunk.get("score", 0)
        context_parts.append(
            f"[Source {i}] (Relevance: {score:.2f}) — {source}\n"
            f"URL: {url}\n"
            f"{chunk['text']}\n"
        )

    return "\n---\n".join(context_parts)
