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
    - Easy to swap components (e.g., replace basic retriever with multi-query)
    - This is the "Strategy Pattern" — change behavior by changing components

DATA FLOW (Stage 3):
    User Question
        ↓
    [if multi_query=True]
        ↓
    LLM generates 3 alternative questions
        ↓
    Retrieve chunks for ALL 4 questions
        ↓
    Deduplicate by chunk ID (keep highest score)
        ↓
    [else: basic retriever with single query]
        ↓
    Format context string with numbered sources
        ↓
    LLM generates grounded answer with citations
        ↓
    Return answer + sources + timing

STAGE 3 UPGRADE:
    Added `retrieval_method` parameter to switch between:
    - "basic" (Stage 1-2): single query, single retrieval
    - "multi_query" (Stage 3): 4 queries, merged retrieval

    The pipeline is BACKWARDS COMPATIBLE — default is "basic".
    You can gradually upgrade without breaking existing code.

INTERVIEW QUESTION:
    "Walk me through your RAG pipeline end-to-end."
    → "The user submits a query. In multi-query mode, I first use
       Gemini to generate 3 alternative phrasings. I embed all 4
       queries and retrieve the top-10 chunks for each from Qdrant.
       After deduplication (keeping highest score per chunk), I format
       the top-10 unique chunks with source numbers and send them
       as context to Gemini. The system prompt instructs the LLM to
       only use provided context and cite sources with [Source N]."
=============================================================
"""

import time
from src.retrieval.basic_retriever import retrieve, format_context
from src.retrieval.multi_query_retriever import multi_query_retrieve
from src.generation.generator import generate_answer


def query_rag(
    question: str,
    top_k: int = 10,
    retrieval_method: str = "basic",
    source_type: str | None = None,
    company_name: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full RAG pipeline for a user question.

    This is the main entry point for the entire system.
    Everything flows through this function.

    RETRIEVAL METHODS:
        "basic" (default):
            Single query → single retrieval. Fast, simple.
            Best for: straightforward factual questions.

        "multi_query" (Stage 3):
            Generate 3 query variants → 4 retrievals → merge.
            Better recall, catches more relevant documents.
            Best for: complex questions, unfamiliar phrasing.

    STAGE 2 FILTERS (passed through):
        source_type: "wikipedia", "yourstory", etc.
        company_name: "Zerodha", "Flipkart", etc.

    Args:
        question: The user's natural language question
        top_k: Number of chunks to retrieve
        retrieval_method: "basic" or "multi_query"
        source_type: Optional metadata filter (Stage 2)
        company_name: Optional metadata filter (Stage 2)
        verbose: If True, print detailed logs

    Returns:
        dict containing:
            - answer: The generated answer text
            - sources: List of source chunks with metadata
            - retrieval_time: Time taken for retrieval (seconds)
            - generation_time: Time taken for generation (seconds)
            - total_time: End-to-end latency
            - model: Which LLM was used
            - retrieval_method: Which retrieval strategy was used
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {question}")
        print(f"   Method: {retrieval_method}")
        if source_type or company_name:
            filters = []
            if source_type:
                filters.append(f"source={source_type}")
            if company_name:
                filters.append(f"company={company_name}")
            print(f"   Filters: {', '.join(filters)}")
        print(f"{'='*60}")

    total_start = time.time()

    # ── Step 1: Retrieve relevant chunks ────────────────────
    retrieval_start = time.time()

    if retrieval_method == "multi_query":
        # Stage 3: Multi-query retrieval
        chunks = multi_query_retrieve(
            question,
            num_queries=3,
            top_k=top_k,
            source_type=source_type,
            company_name=company_name,
            verbose=verbose,
        )
    else:
        # Stage 1-2: Basic single-query retrieval
        chunks = retrieve(
            question,
            top_k=top_k,
            source_type=source_type,
            company_name=company_name,
        )

    retrieval_time = time.time() - retrieval_start

    if verbose:
        print(f"\n📚 Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        for i, chunk in enumerate(chunks, 1):
            company = chunk['metadata'].get('company_name', chunk['metadata'].get('title', '?'))
            print(f"   [{i}] Score: {chunk['score']:.3f} | {company[:50]}")

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the knowledge base.",
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0,
            "total_time": time.time() - total_start,
            "model": "N/A",
            "retrieval_method": retrieval_method,
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
                "title": c["metadata"].get("title", "Unknown"),
                "company": c["metadata"].get("company_name", "Unknown"),
                "url": c["metadata"].get("url", ""),
                "score": c["score"],
                "text_preview": c["text"][:200] + "...",
            }
            for c in chunks
        ],
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(generation_time, 3),
        "total_time": round(total_time, 3),
        "model": result["model"],
        "retrieval_method": retrieval_method,
    }
