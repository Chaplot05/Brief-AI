"""
=============================================================
BASIC_RETRIEVER.PY — Vector Similarity Search
=============================================================

WHAT THIS DOES:
    Takes a user query, embeds it, and finds the most similar
    chunks in ChromaDB using cosine similarity.

THIS IS THE "R" IN RAG:
    RAG = Retrieval-Augmented Generation
    - Retrieval: find relevant documents (THIS FILE)
    - Augmented: add those documents to the LLM's context
    - Generation: LLM generates an answer using the context

HOW RETRIEVAL WORKS:
    1. User asks: "Who founded Zerodha?"
    2. We embed the query → get a 768-dim vector
    3. ChromaDB finds the 10 chunks whose vectors are closest
       to the query vector (using cosine similarity)
    4. We return those chunks as "context" for the LLM

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

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)


def retrieve(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    THE RETRIEVAL PIPELINE:
        1. Embed the query using the SAME embedding model used for indexing
           (CRITICAL: query and documents MUST use the same model,
            otherwise the vector spaces don't align)
        2. ChromaDB computes cosine similarity between query vector
           and ALL stored vectors
        3. Returns the top-k most similar chunks

    WHAT CHROMADB RETURNS:
        - documents: the original text of each chunk
        - metadatas: article title, URL, etc.
        - distances: similarity scores (lower = more similar for cosine)
        - ids: unique chunk identifiers

    Args:
        query: The user's question
        top_k: Number of chunks to retrieve

    Returns:
        List of dicts, each containing:
            - text: chunk text
            - metadata: article info (title, URL, etc.)
            - score: similarity score
            - id: chunk identifier
    """
    # Initialize embedding function (SAME model as used in indexing)
    embedding_fn = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # Embed the query
    query_embedding = embedding_fn.embed_query(query)

    # Connect to ChromaDB and get the collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    # Query ChromaDB for the top-k most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Format results into a clean list of dicts
    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            # ChromaDB returns "distance" — for cosine, lower = more similar
            # We convert to a similarity score: 1 - distance
            "score": 1 - results["distances"][0][i],
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
