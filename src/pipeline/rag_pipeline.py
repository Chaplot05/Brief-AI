"""
=============================================================
RAG_PIPELINE.PY — Full RAG Pipeline Orchestrator
=============================================================

WHAT THIS DOES:
    Wires together all the components:
    Query → Retrieve → Format Context → Generate → Return

WHY AN ORCHESTRATOR?
    - Each component (retriever, generator) is independent and testable
    - The pipeline connects them into a complete flow
    - Easy to swap components (e.g., replace basic retriever with HyDE)
    - This is the "Strategy Pattern" — change behavior by changing components

DATA FLOW:
    User Question
        ↓
    basic_retriever.retrieve(question)
        ↓
    List of relevant chunks (with scores + metadata)
        ↓
    basic_retriever.format_context(chunks)
        ↓
    Formatted context string
        ↓
    generator.generate_answer(question, context)
        ↓
    Answer with citations

INTERVIEW QUESTION:
    "Walk me through your RAG pipeline end-to-end."
    → "The user submits a query. I embed it using Gemini embeddings
       and retrieve the top-10 most similar chunks from ChromaDB using
       cosine similarity. I format those chunks with source numbers
       and send them as context to Gemini along with the question.
       The system prompt instructs the LLM to only use the provided
       context and cite sources. The response includes the answer
       and source citations."
=============================================================
"""

import time
from src.retrieval.basic_retriever import retrieve, format_context
from src.generation.generator import generate_answer


def query_rag(
    question: str,
    top_k: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run the full RAG pipeline for a user question.

    This is the main entry point for the entire system.
    Everything flows through this function.

    Args:
        question: The user's natural language question
        top_k: Number of chunks to retrieve
        verbose: If True, print detailed logs

    Returns:
        dict containing:
            - answer: The generated answer text
            - sources: List of source chunks with metadata
            - retrieval_time: Time taken for retrieval (seconds)
            - generation_time: Time taken for generation (seconds)
            - total_time: End-to-end latency
            - model: Which LLM was used
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {question}")
        print(f"{'='*60}")

    total_start = time.time()

    # ── Step 1: Retrieve relevant chunks ────────────────────
    retrieval_start = time.time()
    chunks = retrieve(question, top_k=top_k)
    retrieval_time = time.time() - retrieval_start

    if verbose:
        print(f"\n📚 Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        for i, chunk in enumerate(chunks, 1):
            print(f"   [{i}] Score: {chunk['score']:.3f} | {chunk['metadata']['title'][:50]}")

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the knowledge base.",
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0,
            "total_time": time.time() - total_start,
            "model": "N/A",
        }

    # ── Step 2: Format context for the LLM ─────────────────
    context = format_context(chunks)

    # ── Step 3: Generate answer ─────────────────────────────
    generation_start = time.time()
    result = generate_answer(question, context)
    generation_time = time.time() - generation_start

    if verbose:
        print(f"\n💡 Answer generated in {generation_time:.2f}s")
        print(f"\n{'─'*60}")
        print(result["answer"])
        print(f"{'─'*60}")

    total_time = time.time() - total_start

    return {
        "answer": result["answer"],
        "sources": [
            {
                "title": c["metadata"]["title"],
                "url": c["metadata"]["url"],
                "score": c["score"],
                "text_preview": c["text"][:200] + "...",
            }
            for c in chunks
        ],
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(generation_time, 3),
        "total_time": round(total_time, 3),
        "model": result["model"],
        "retrieval_method": "basic_vector_search",
    }
