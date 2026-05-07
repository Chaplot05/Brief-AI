"""
=============================================================
CHUNKER.PY — Text Chunking with Overlap
=============================================================

WHAT THIS DOES:
    Takes full article text and splits it into smaller "chunks"
    that can be embedded and retrieved independently.

WHY WE NEED CHUNKING:
    1. Embedding models have input limits (~8K tokens for Gemini)
    2. LLMs have context window limits
    3. MOST IMPORTANTLY: smaller chunks = more precise retrieval

    Imagine asking "Who founded Zerodha?"
    - A 5000-token article about Zerodha covers founding, revenue,
      products, team, office, competition...
    - A 400-token chunk about Zerodha's founding contains EXACTLY
      the answer with minimal noise
    - The LLM gets a focused context → better answer

WHY 400 TOKENS (not 200 or 1000)?
    - 200 tokens: too small, loses context (a paragraph about funding
      might be split across 3 chunks, each incomplete)
    - 400 tokens: sweet spot — ~1-2 paragraphs, enough context to
      be self-contained, small enough for precise retrieval
    - 1000 tokens: too large, retrieval returns "approximately relevant"
      chunks with lots of noise
    - Industry standard: 200-500 tokens for most RAG systems

WHY OVERLAP?
    Consider this text split at position 400:
      Chunk 1: "...Zerodha was founded in"
      Chunk 2: "2010 by Nithin Kamath..."

    Without overlap, the founding fact is SPLIT between two chunks.
    Neither chunk contains the complete information.

    With 50-token overlap:
      Chunk 1: "...Zerodha was founded in 2010 by Nithin Kamath..."
      Chunk 2: "...was founded in 2010 by Nithin Kamath. The company..."

    Both chunks now contain the complete sentence.
    50 tokens ≈ 2-3 sentences of overlap — enough to preserve context.

INTERVIEW QUESTION:
    "How did you decide on chunk size?"
    → "I experimented with 200, 400, and 600 token chunks. 400 gave
       the best retrieval precision because chunks were self-contained
       but focused. I used 50-token overlap to prevent information
       loss at boundaries."

WHAT COMPANIES DO:
    - Anthropic/OpenAI recommend 200-500 token chunks
    - Some use "semantic chunking" (split at paragraph/section boundaries)
    - Some use "recursive" chunking (try to split at \n\n, then \n, then space)
    - We'll upgrade to recursive chunking in Stage 2
=============================================================
"""

import tiktoken
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a text string.

    WHY COUNT TOKENS INSTEAD OF CHARACTERS?
        - LLMs and embedding models think in TOKENS, not characters
        - "Nithin Kamath" = 2 words, 14 chars, but might be 3-4 tokens
        - Token count determines:
          1. Whether text fits in the model's input limit
          2. How much "space" a chunk takes in the context window
          3. API costs (charged per token)

    WHAT IS A TOKENIZER?
        - Converts text → list of token IDs
        - "I love AI" → [40, 3476, 15836] (3 tokens)
        - Different models use different tokenizers
        - cl100k_base is used by GPT-4 and is a good general-purpose counter

    Args:
        text: The text to count tokens for
        model: The tokenizer encoding to use

    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks based on token count.

    ALGORITHM:
        1. Split text into sentences (using period + space as delimiter)
        2. Accumulate sentences until we reach chunk_size tokens
        3. Save that chunk, then back up by chunk_overlap tokens
        4. Continue from the overlap point

    WHY SENTENCE-BASED SPLITTING?
        - Splitting mid-sentence creates garbage chunks
        - "Zerodha raised $" and "500M in Series B" are useless alone
        - Sentence boundaries create coherent, meaningful chunks

    Args:
        text: The full text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Tokens to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split into sentences (simple but effective)
    # In Stage 2, we'll use LangChain's RecursiveCharacterTextSplitter
    # which tries \n\n → \n → ". " → " " in order
    sentences = text.replace("\n", " ").split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # If adding this sentence would exceed chunk_size,
        # save the current chunk and start a new one
        if current_token_count + sentence_tokens > chunk_size and current_chunk_sentences:
            # Save the completed chunk
            chunk_text_str = " ".join(current_chunk_sentences)
            chunks.append(chunk_text_str)

            # OVERLAP: Keep some sentences from the end of this chunk
            # to start the next chunk. This is the key overlap mechanism.
            overlap_sentences = []
            overlap_tokens = 0
            for sent in reversed(current_chunk_sentences):
                sent_tokens = count_tokens(sent)
                if overlap_tokens + sent_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Don't forget the last chunk!
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences)
        if count_tokens(chunk_text_str) > 30:  # Skip tiny trailing chunks
            chunks.append(chunk_text_str)

    return chunks


def chunk_articles(articles: list[dict]) -> list[dict]:
    """
    Chunk all articles and attach metadata to each chunk.

    WHY ATTACH METADATA?
        - Each chunk needs to "remember" which article it came from
        - This enables CITATIONS in the final answer
        - Without metadata, you'd show the answer but couldn't say
          "Source: YourStory article from 2024"
        - Citations build user trust and are a MUST for production RAG

    WHAT METADATA WE ATTACH:
        - article_id: unique identifier
        - title: article title (for display)
        - url: source URL (for citation links)
        - chunk_index: position within the article (for ordering)

    Args:
        articles: List of article dicts from the scraper

    Returns:
        List of chunk dicts with text + metadata
    """
    all_chunks = []

    for article in articles:
        text = article.get("text", "")
        if not text:
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{article['id']}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "article_id": article["id"],
                    "title": article.get("title", "Unknown"),
                    "url": article.get("url", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "token_count": count_tokens(chunk),
                },
            })

    print(f"📦 Created {len(all_chunks)} chunks from {len(articles)} articles")
    print(f"   Average chunk size: {sum(c['metadata']['token_count'] for c in all_chunks) / max(len(all_chunks), 1):.0f} tokens")
    return all_chunks
