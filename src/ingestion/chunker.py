"""
=============================================================
CHUNKER.PY — Stage 2: RecursiveCharacterTextSplitter
=============================================================

WHAT CHANGED FROM STAGE 1:
    Stage 1 used a SIMPLE sentence-based splitter:
        text.split(". ")  →  join sentences until 400 tokens

    Stage 2 upgrades to LangChain's RecursiveCharacterTextSplitter:
        Try splitting at: \\n\\n → \\n → . → " " → ""
        This preserves document STRUCTURE, not just sentences.

WHY THIS MATTERS — THE CORE PROBLEM WITH STAGE 1:

    Stage 1's splitter had a critical flaw:
        text.replace("\\n", " ").split(". ")

    This DESTROYS ALL paragraph and section structure:
    - Wikipedia articles have headers: "== History =="
    - Paragraphs are separated by \\n\\n
    - Our Stage 1 splitter collapsed everything into one long string
    - Then blindly split on periods

    Example of what went wrong:
        Original text:
            == History ==
            Zerodha was founded in 2010 by Nithin Kamath.

            == Products ==
            Zerodha offers Kite, a trading platform.

        Stage 1 output (BAD):
            Chunk: "== History == Zerodha was founded in 2010 by Nithin Kamath."
            → The section header is GLUED to the text. No structure preserved.

        Stage 2 output (GOOD):
            Chunk: "History\\nZerodha was founded in 2010 by Nithin Kamath."
            → Paragraph boundaries preserved. Context is cleaner.

WHAT IS RecursiveCharacterTextSplitter?

    It's LangChain's smartest text splitter. The "recursive" means it
    tries MULTIPLE separators in order of preference:

    1. First try: split on "\\n\\n" (paragraph boundaries)
       → Keeps entire paragraphs together. Best quality chunks.
       → But if a paragraph is > 400 tokens, it won't fit.

    2. Fallback: split on "\\n" (line breaks)
       → Keeps lines together. Still decent structure.

    3. Fallback: split on ". " (sentence boundaries)
       → Same as our Stage 1 approach. Sentences stay whole.

    4. Last resort: split on " " (word boundaries)
       → Never splits mid-word. Guaranteed coherent text.

    This hierarchy means:
    - SHORT paragraphs → stay together as one chunk
    - LONG paragraphs → split at sentence boundaries
    - VERY long sentences → split at word boundaries

    WHY IS THIS BETTER?
    Because paragraph = one idea. Keeping paragraphs intact means
    each chunk contains ONE complete thought, not fragments of two.

WHY from_tiktoken_encoder() ?

    RecursiveCharacterTextSplitter normally counts CHARACTERS.
    But LLMs think in TOKENS. 400 characters ≠ 400 tokens.

    from_tiktoken_encoder() makes it count TOKENS instead:
    - chunk_size=400 means 400 TOKENS (not characters)
    - This matches our Stage 1 behavior
    - More accurate for LLM context window budgeting

    TRADEOFF: Token counting is ~10x slower than character counting.
    But with 27 articles, this takes seconds either way. In production
    with millions of documents, you might use character counting for
    speed and accept the slight inaccuracy.

WHAT ELSE IS NEW IN STAGE 2:

    1. METADATA ENRICHMENT:
       - Added "source_type" field (wikipedia, yourstory, etc.)
       - Extracted company/topic names from titles
       - These enable FILTERED search in Qdrant

    2. CHUNK QUALITY METRICS:
       - Track min/max/avg/median chunk sizes
       - This helps diagnose chunking issues

INTERVIEW QUESTION:
    "How did you improve your chunking strategy?"
    → "I started with simple sentence splitting and upgraded to
       LangChain's RecursiveCharacterTextSplitter. It tries paragraph
       boundaries first, then sentences, then words. This keeps
       semantic structure intact — paragraphs about one topic stay
       in one chunk instead of being fragmented across multiple chunks.
       I also enriched metadata with source types for filtered search."

    "What's the difference between character-based and token-based chunking?"
    → "LLMs process tokens, not characters. 'Nithin Kamath' is 14
       characters but 3-4 tokens. Using token-based chunking ensures
       each chunk fits precisely in the model's context window. I use
       tiktoken's cl100k_base encoding for accurate token counting."
=============================================================
"""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Token Counter (kept from Stage 1) ──────────────────────
# Still needed for metadata and quality metrics.

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


# ── The Splitter — Stage 2's Main Upgrade ──────────────────

def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with token-based sizing.

    WHY A FACTORY FUNCTION?
        - Makes it easy to test different configurations
        - In Stage 3 (multi-query), we might want different chunk sizes
        - Factory pattern = create objects through a function, not directly

    HOW from_tiktoken_encoder() WORKS:
        Internally, it wraps the splitter so that:
        1. chunk_size and chunk_overlap are in TOKENS (not characters)
        2. When deciding where to split, it counts tokens using tiktoken
        3. The separators list still defines WHERE to try splitting
        4. But the SIZE constraints are in tokens

    THE SEPARATOR HIERARCHY (most preferred → least preferred):
        "\\n\\n" — Paragraph boundary (best: keeps paragraphs whole)
        "\\n"   — Line break (good: keeps lines together)
        ". "   — Sentence boundary (okay: keeps sentences whole)
        " "    — Word boundary (fallback: never splits mid-word)
        ""     — Character boundary (last resort: never actually used)

    Args:
        chunk_size: Maximum tokens per chunk (default: 400)
        chunk_overlap: Overlap tokens between chunks (default: 50)

    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",   # Same tokenizer as count_tokens()
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        # strip_whitespace: removes leading/trailing whitespace from chunks
        # Important because after splitting on \n\n, chunks may start with \n
        strip_whitespace=True,
        # is_separator_regex=False means separators are treated as literal strings
        # Set to True if you want regex patterns (e.g., r"\n{2,}" for 2+ newlines)
        is_separator_regex=False,
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.

    STAGE 1 vs STAGE 2 — WHAT CHANGED:
        Stage 1:
            text.replace("\\n", " ").split(". ")
            → Destroyed all structure. Simple but lossy.

        Stage 2:
            RecursiveCharacterTextSplitter(separators=[...])
            → Preserves paragraph and section structure.
            → Tries smart boundaries first, falls back gracefully.

    Args:
        text: The full text to chunk
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Tokens to overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    splitter = get_text_splitter(chunk_size, chunk_overlap)

    # split_text() applies the recursive splitting algorithm:
    # 1. Try to split on \n\n (paragraphs)
    # 2. If a paragraph > chunk_size tokens, split it on \n (lines)
    # 3. If a line > chunk_size tokens, split on ". " (sentences)
    # 4. If a sentence > chunk_size tokens, split on " " (words)
    # Then it merges small consecutive pieces back together with overlap
    chunks = splitter.split_text(text)

    # Filter out tiny chunks (< 30 tokens) — they add noise without information
    # This can happen with short paragraphs at the end of articles
    chunks = [c for c in chunks if count_tokens(c) > 30]

    return chunks


def _extract_source_type(url: str) -> str:
    """
    Determine the source type from the article URL.

    WHY TRACK SOURCE TYPE?
        In Qdrant, payload fields can be INDEXED and used for
        FILTERED SEARCH. Example queries:
        - "Find startup funding info, but ONLY from Wikipedia articles"
        - "Search for product info, but ONLY from company websites"

        This is a huge advantage over ChromaDB, which has limited
        filtering capabilities. Qdrant can combine:
        1. Vector similarity (semantic search)
        2. Payload filters (metadata constraints)

        This is called HYBRID FILTERING — you get the best of both
        SQL (exact filters) and vector search (semantic similarity).

    INTERVIEW QUESTION:
        "How do you filter search results in your RAG system?"
        → "Qdrant supports payload-based filtering alongside vector
           search. I index metadata fields like source_type and
           company_name, so I can run queries like 'find the top-5
           semantically similar chunks, but only from Wikipedia sources'.
           This reduces noise and improves retrieval precision."

    Args:
        url: Article URL

    Returns:
        Source type string (e.g., "wikipedia", "yourstory")
    """
    url_lower = url.lower()
    if "wikipedia.org" in url_lower:
        return "wikipedia"
    elif "yourstory.com" in url_lower:
        return "yourstory"
    elif "inc42.com" in url_lower:
        return "inc42"
    elif "techcrunch.com" in url_lower:
        return "techcrunch"
    elif "economictimes" in url_lower:
        return "economic_times"
    else:
        return "other"


def _extract_company_name(title: str) -> str:
    """
    Extract a clean company/topic name from the article title.

    Wikipedia titles often look like:
        "Zerodha - Wikipedia"
        "Flipkart"
        "Unified Payments Interface"

    We clean these up to use as filterable metadata.

    Args:
        title: Article title

    Returns:
        Cleaned company/topic name
    """
    # Remove common suffixes
    name = title
    for suffix in [" - Wikipedia", " – Wikipedia", " | Wikipedia"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    # Remove "(company)" type suffixes for cleaner names
    if name.endswith(")"):
        paren_start = name.rfind("(")
        if paren_start > 0:
            name = name[:paren_start].strip()

    return name.strip()


def chunk_articles(articles: list[dict]) -> list[dict]:
    """
    Chunk all articles and attach enriched metadata to each chunk.

    STAGE 2 IMPROVEMENTS:
        1. Uses RecursiveCharacterTextSplitter (better boundaries)
        2. Adds source_type metadata (for filtered search)
        3. Adds company_name metadata (for filtered search)
        4. Reports quality metrics (min/max/avg chunk size)

    WHY ENRICHED METADATA?
        Stage 1 metadata: {title, url, article_id, chunk_index}
        Stage 2 metadata: + {source_type, company_name, total_tokens}

        The new fields enable Qdrant payload filtering:
        - Filter by source: "only Wikipedia articles"
        - Filter by company: "only Zerodha chunks"
        - Filter by size: "only chunks > 100 tokens" (quality filter)

    Args:
        articles: List of article dicts from the scraper

    Returns:
        List of chunk dicts with text + enriched metadata
    """
    all_chunks = []
    chunk_sizes = []  # Track sizes for quality metrics

    for article in articles:
        text = article.get("text", "")
        if not text:
            continue

        url = article.get("url", "")
        title = article.get("title", "Unknown")

        # ── Stage 2 metadata enrichment ────────────────────
        source_type = _extract_source_type(url)
        company_name = _extract_company_name(title)

        # ── Chunk with RecursiveCharacterTextSplitter ──────
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            token_count = count_tokens(chunk)
            chunk_sizes.append(token_count)

            all_chunks.append({
                "id": f"{article['id']}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    # ── Original metadata (Stage 1) ────────
                    "article_id": article["id"],
                    "title": title,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    # ── New metadata (Stage 2) ─────────────
                    "source_type": source_type,
                    "company_name": company_name,
                },
            })

    # ── Quality Metrics ────────────────────────────────────
    if chunk_sizes:
        sorted_sizes = sorted(chunk_sizes)
        median = sorted_sizes[len(sorted_sizes) // 2]
        print(f"📦 Created {len(all_chunks)} chunks from {len(articles)} articles")
        print(f"   Splitter:   RecursiveCharacterTextSplitter (Stage 2)")
        print(f"   Avg size:   {sum(chunk_sizes) / len(chunk_sizes):.0f} tokens")
        print(f"   Median:     {median} tokens")
        print(f"   Range:      {min(chunk_sizes)} – {max(chunk_sizes)} tokens")
        print(f"   Config:     chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    else:
        print(f"⚠ No chunks created from {len(articles)} articles")

    return all_chunks
"""

"""
