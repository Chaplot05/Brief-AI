"""
=============================================================
GENERATOR.PY — LLM Answer Generation with Citations
=============================================================

WHAT THIS DOES:
    Takes retrieved chunks (context) and the user's question,
    sends them to Gemini, and generates a grounded answer.

THIS IS THE "G" IN RAG:
    - The LLM doesn't answer from its training data (parametric memory)
    - Instead, it answers ONLY from the provided context (non-parametric)
    - This is called "grounding" — the answer is grounded in real data

WHY GROUNDING MATTERS:
    Without grounding (plain LLM):
        Q: "How much funding did Zerodha raise?"
        A: "Zerodha raised $500M in Series C" ← HALLUCINATION (made up!)

    With grounding (RAG):
        Q: "How much funding did Zerodha raise?"
        Context: [chunk about Zerodha being bootstrapped]
        A: "Zerodha is bootstrapped and hasn't raised external funding."
        ← CORRECT, because the LLM was forced to use the context

WHY CITATIONS MATTER:
    - Users need to verify claims ("trust but verify")
    - Perplexity AI shows citations for every claim
    - Without citations, your RAG is just a chatbot
    - Citations are what make RAG TRUSTWORTHY

THE SYSTEM PROMPT:
    The system prompt is CRITICAL in RAG. It tells the LLM:
    1. "Only answer from the provided context"
    2. "Say 'I don't know' if the context doesn't contain the answer"
    3. "Cite your sources using [Source N]"
    These instructions prevent hallucination and ensure grounding.

INTERVIEW QUESTION:
    "How do you prevent hallucinations?"
    → "Three layers: (1) System prompt instructs the LLM to only use
       provided context. (2) Citations force the LLM to point to
       specific sources. (3) Self-correction (Stage 6) verifies the
       answer against the context and flags unsupported claims."
=============================================================
"""

import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import GOOGLE_API_KEY, GENERATION_MODEL


# ── System Prompt ──────────────────────────────────────────
# This is the most important part of the RAG system.
# Every word is intentional.
SYSTEM_PROMPT = """You are an expert analyst on the Indian startup ecosystem.
Your role is to provide accurate, well-researched answers based ONLY on the provided context.

RULES:
1. Answer ONLY based on the provided context documents.
2. If the context doesn't contain enough information to answer, say:
   "I don't have enough information in my knowledge base to answer this question accurately."
3. Cite your sources using [Source N] notation matching the source numbers in the context.
4. Be specific with facts, numbers, and names from the context.
5. If multiple sources provide different information, mention the discrepancy.
6. Keep your answer concise but comprehensive.
7. Do NOT make up information that isn't in the context.

CONTEXT:
{context}

USER QUESTION:
{question}

Provide a well-structured answer with citations:"""


def get_llm():
    """
    Initialize the Gemini LLM for answer generation.

    TEMPERATURE = 0.1:
        - Controls randomness. 0 = deterministic, 1 = creative
        - For RAG, we want FACTUAL answers, not creative ones

    Returns:
        ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )


def generate_answer(question: str, context: str, max_retries: int = 3) -> dict:
    """
    Generate an answer with retry logic for rate limits.

    RETRY WITH EXPONENTIAL BACKOFF:
        API rate limits are a fact of life in production.
        Instead of crashing, we wait and retry:
        - 1st retry: wait 30s
        - 2nd retry: wait 60s
        - 3rd retry: wait 90s
        This is called "exponential backoff" — every production
        system that calls external APIs implements this pattern.

    Args:
        question: The user's question
        context: Formatted context string from retriever
        max_retries: Number of retry attempts

    Returns:
        dict with 'answer' and 'model' keys
    """
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    for attempt in range(max_retries + 1):
        try:
            answer = chain.invoke({
                "question": question,
                "context": context,
            })
            return {
                "answer": answer,
                "model": GENERATION_MODEL,
            }
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait_time = 30 * (attempt + 1)
                print(f"  Rate limited. Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

