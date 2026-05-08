"""
=============================================================
HYDE_RETRIEVER.PY — Stage 4: Hypothetical Document Embeddings
=============================================================

WHAT THIS DOES:
    Instead of embedding the user's QUESTION, we:
    1. Ask the LLM to generate a HYPOTHETICAL answer
    2. Embed that hypothetical answer
    3. Search Qdrant using the answer's embedding

    The hypothetical answer is NEVER shown to the user.
    It's only used as a "search vector."

WHY THIS WORKS — THE KEY INSIGHT:
    There's a fundamental ASYMMETRY in RAG:
    - User queries are SHORT questions: "Who founded Zerodha?"
    - Knowledge base has LONG passages: "Zerodha Broking Ltd is an
      Indian financial services company founded in 2010 by Nithin
      Kamath and Nikhil Kamath in Bangalore..."

    When you embed the question and embed the passage, they live in
    DIFFERENT regions of the embedding space because they look
    different — one is a question, the other is a statement.

    HyDE bridges this gap:
    - Generate a hypothetical answer that LOOKS LIKE a passage
    - Its embedding will be in the SAME region as real passages
    - Better similarity scores → better retrieval

    Think of it as: "To find a book, write a fake summary of what
    the book would say, then search the library for books that
    match your summary."

=============================================================
THE HALLUCINATION QUESTION (What you asked! 🎯)
=============================================================

Q: "What if the LLM generates a WRONG hypothetical answer?"
A: **It usually doesn't matter.** Here's why, step by step:

SCENARIO 1: Factually wrong, topically correct (WORKS FINE)
    User asks: "How much funding has Zerodha raised?"
    
    LLM hallucinates: "Zerodha has raised $500M in Series B funding
    from Sequoia Capital and Tiger Global."
    
    This is FACTUALLY WRONG (Zerodha is bootstrapped!).
    But the embedding captures the VOCABULARY:
    - "Zerodha", "raised", "funding", "Series B", "bootstrapped"
    
    This vocabulary is EXACTLY what appears in real chunks about
    Zerodha's funding. So the search will still find:
    - "Zerodha is one of few profitable bootstrapped startups..."
    - "Unlike other startups, Zerodha never raised VC funding..."
    
    The REAL chunks contain the CORRECT facts.
    The LLM then generates the final answer from REAL chunks only.
    
    Result: User gets the correct answer ("Zerodha is bootstrapped")
    even though HyDE hallucinated.

SCENARIO 2: Completely off-topic hallucination (RARE, BAD)
    User asks: "Tell me about Zerodha"
    
    LLM hallucinates: "Zerodha is a popular food delivery app
    that competes with Swiggy and Zomato."
    
    Now the embedding is about "food delivery" — WRONG topic.
    It will match chunks about Swiggy/Zomato, not Zerodha.
    
    This is the REAL risk of HyDE. But it's rare because:
    1. We keep hypothetical answers SHORT (2-3 sentences)
    2. We use low temperature (0.3) for more focused output
    3. The question itself anchors the topic
    4. Modern LLMs rarely hallucinate THIS badly

SCENARIO 3: Partial hallucination (STILL WORKS)
    User asks: "What products does Flipkart offer?"
    
    LLM hallucinates: "Flipkart offers electronics, fashion,
    groceries, and its own payment system FlipPay."
    
    "FlipPay" doesn't exist, but "electronics", "fashion",
    "groceries" are correct. The embedding will be 80% correct,
    which is still MUCH better than embedding the question
    "What products does Flipkart offer?" alone.

MITIGATIONS WE IMPLEMENT:
    1. SHORT hypothetical answers (2-3 sentences max)
       → Less room for the LLM to go off-topic
    2. LOW temperature (0.3)
       → More focused, less creative = less hallucination
    3. COMBINE with original query embedding
       → We search with BOTH the question AND the hypothesis
       → If HyDE's embedding is bad, the original still works
    4. CORRECTIVE RAG (Stage 6+) catches retrieval errors anyway

INTERVIEW ANSWER:
    "Aren't you worried about HyDE hallucinating?"
    → "Great question. HyDE uses the hypothetical answer only for its
       VOCABULARY, not its FACTS. Even a factually wrong hypothesis
       about 'Zerodha funding' will use the right domain words that
       match real documents. The actual answer is always generated
       from grounded knowledge base chunks, never from the hypothesis.
       I also mitigate by keeping hypotheses short, using low temperature,
       and combining HyDE with the original query embedding as a fallback."
=============================================================
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import (
    GOOGLE_API_KEY,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)


# ── HyDE Prompt ──────────────────────────────────────────────
# This prompt is carefully designed to:
# 1. Generate a SHORT answer (2-3 sentences) → less hallucination
# 2. Write in DOCUMENT STYLE (not conversational) → better embedding match
# 3. Focus on the TOPIC (not accuracy) → right vocabulary
# 4. Include domain context (Indian startups) → grounded vocabulary
#
# WHY "Write as if you are writing a Wikipedia article"?
#   Our knowledge base IS Wikipedia articles. By asking the LLM
#   to write in Wikipedia style, the hypothetical answer will use
#   the same writing patterns, making its embedding align better
#   with the real Wikipedia chunks in Qdrant.
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """You are writing a short passage for a knowledge base about the Indian startup ecosystem.

Given this question, write a brief 2-3 sentence passage that would answer it.
Write as if you are writing a Wikipedia article — factual, neutral tone.
Focus on being topically relevant, using domain-specific vocabulary.

Do NOT say "I don't know" or "Based on my knowledge." Just write the passage directly.

Question: {question}

Passage:"""
)


def generate_hypothetical_answer(
    question: str,
    max_retries: int = 3,
) -> str | None:
    """
    Generate a hypothetical answer to the user's question.

    This answer may contain hallucinations — that's OKAY.
    We only use it for its VOCABULARY, not its FACTS.

    WHY temperature=0.3?
        - 0.0: Too deterministic, always generates the same text
        - 0.3: Slightly creative, but focused on the topic
        - 0.7: Too creative, higher risk of going off-topic
        - 1.0: Very creative, HIGH risk of off-topic hallucination

        For HyDE, we want FOCUSED, not CREATIVE.
        We want the right words, not interesting prose.

    Args:
        question: The user's original question

    Returns:
        Hypothetical answer string, or None if generation fails
    """
    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,  # Low temp = focused, less hallucination
    )
    chain = HYDE_PROMPT | llm | StrOutputParser()

    for attempt in range(max_retries + 1):
        try:
            hypothesis = chain.invoke({"question": question})
            return hypothesis.strip()
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries:
                wait_time = 30 * (attempt + 1)
                print(f"  ⏳ Rate limited during HyDE generation. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠ HyDE generation failed: {e}")
                return None


def hyde_retrieve(
    question: str,
    top_k: int = TOP_K_RETRIEVAL,
    source_type: str | None = None,
    company_name: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Retrieve chunks using Hypothetical Document Embeddings (HyDE).

    THE HYDE ALGORITHM:
        1. Generate a hypothetical answer using the LLM
        2. Embed the hypothetical answer (NOT the question)
        3. ALSO embed the original question (as fallback)
        4. Search Qdrant with BOTH embeddings
        5. Merge results, deduplicate by chunk ID
        6. Return top-k unique chunks

    WHY SEARCH WITH BOTH (hypothesis + original)?
        This is our SAFETY NET against bad hallucinations.
        - If HyDE generates a good hypothesis → great results
        - If HyDE generates garbage → original query still works
        - We get the UNION of both result sets, deduped by max score

        This is sometimes called "ensemble retrieval" — combining
        multiple retrieval strategies for robustness.

    COMPARISON OF RETRIEVAL STRATEGIES:
        Basic (Stage 1-2):  embed(question) → search
        Multi-Query (Stage 3): embed(question + 3 variants) → search
        HyDE (Stage 4):  embed(hypothesis + question) → search

        Each catches different types of relevant documents:
        - Basic: direct question-document matches
        - Multi-Query: varied phrasing matches
        - HyDE: document-style vocabulary matches

    Args:
        question: User's original question
        top_k: Number of chunks to return
        source_type: Optional metadata filter
        company_name: Optional metadata filter
        verbose: Print detailed logs

    Returns:
        List of deduplicated, top-scoring chunk dicts
    """
    # ── Step 1: Generate hypothetical answer ───────────────
    if verbose:
        print(f"  🧪 Generating hypothetical answer (HyDE)...")

    hypothesis = generate_hypothetical_answer(question)

    if hypothesis and verbose:
        # Show first 150 chars of the hypothesis
        preview = hypothesis[:150].replace("\n", " ")
        print(f"  📝 Hypothesis: \"{preview}...\"")

    # ── Step 2: Build embeddings ───────────────────────────
    embedding_fn = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # Always embed the original question (safety net)
    query_embedding = embedding_fn.embed_query(question)

    # Embed hypothesis if available
    hyde_embedding = None
    if hypothesis:
        hyde_embedding = embedding_fn.embed_query(hypothesis)

    # ── Step 3: Search with both embeddings ────────────────
    client = QdrantClient(path=QDRANT_PATH)

    # Build filter (same as Stage 2)
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

    # Search with original question
    all_chunks = {}  # chunk_id → chunk dict (dedup map)

    original_results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    if verbose:
        print(f"    Original query: {len(original_results.points)} results")

    INTERNAL_FIELDS = {"text", "chunk_id"}
    for point in original_results.points:
        chunk_id = point.payload.get("chunk_id", str(point.id))
        metadata = {k: v for k, v in point.payload.items() if k not in INTERNAL_FIELDS}
        all_chunks[chunk_id] = {
            "id": chunk_id,
            "text": point.payload["text"],
            "metadata": metadata,
            "score": point.score,
        }

    # Search with hypothesis embedding
    if hyde_embedding:
        hyde_results = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=hyde_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        if verbose:
            print(f"    HyDE query:     {len(hyde_results.points)} results")

        for point in hyde_results.points:
            chunk_id = point.payload.get("chunk_id", str(point.id))
            if chunk_id not in all_chunks or point.score > all_chunks[chunk_id]["score"]:
                metadata = {k: v for k, v in point.payload.items() if k not in INTERNAL_FIELDS}
                all_chunks[chunk_id] = {
                    "id": chunk_id,
                    "text": point.payload["text"],
                    "metadata": metadata,
                    "score": point.score,
                }

    # ── Step 4: Sort and return top-k ──────────────────────
    sorted_chunks = sorted(
        all_chunks.values(),
        key=lambda c: c["score"],
        reverse=True,
    )
    final_chunks = sorted_chunks[:top_k]

    if verbose:
        print(f"\n  📊 Results: {len(all_chunks)} unique chunks (original + HyDE)")
        print(f"  📊 Returning top {len(final_chunks)} by score")

    return final_chunks
