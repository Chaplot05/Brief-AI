"""
=============================================================
CONFIG.PY — Centralized Configuration
=============================================================

WHY THIS FILE EXISTS:
    In production systems, you NEVER hardcode values like API keys,
    model names, or chunk sizes directly in your code. Instead, you
    put them in environment variables and load them in ONE place.

WHY ONE PLACE?
    - If you want to change the chunk size from 400 to 500, you change
      it HERE, not in 15 different files.
    - This is called the "Single Source of Truth" pattern.
    - Every company does this. Interviewers will ask about config management.

WHY ENVIRONMENT VARIABLES?
    - API keys must NEVER be in source code (security risk)
    - Different environments (dev, staging, prod) need different values
    - .env files are gitignored, so keys never leak to GitHub

LIBRARY: python-dotenv
    - Reads key=value pairs from a .env file
    - Loads them into os.environ so your code can access them
    - Alternative: use Docker secrets or cloud secret managers in production
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ──────────────────────────────────────────
# load_dotenv() searches for a .env file starting from the current
# directory and walking up. It loads all key=value pairs into the
# process's environment variables.
load_dotenv()

# ── Project Paths ───────────────────────────────────────────
# Path(__file__) = path to THIS file (config.py)
# .parent = src/
# .parent = project root
# This makes all paths relative to the project root, so the code
# works regardless of where you run it from.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ── Google Gemini API ───────────────────────────────────────
# We use Google's Gemini API for TWO things:
# 1. Embeddings: converting text → vectors (gemini-embedding-001)
# 2. Generation: producing answers from context (gemini-2.0-flash)
#
# WHY GEMINI?
# - Free tier is generous (enough for this project)
# - gemini-embedding-001 produces 768-dim vectors (good quality)
# - gemini-2.0-flash is fast and capable
# - Alternative: OpenAI (costs money), Sentence Transformers (local but slower)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.0-flash")

# ── Embedding Dimensions ──────────────────────────────────
# WHY DO WE NEED THIS?
#   ChromaDB inferred the vector size automatically when you first
#   inserted data. Qdrant is more explicit — you MUST declare the
#   vector size when creating a collection.
#
# WHY 3072?
#   Google's gemini-embedding-001 outputs 3072-dimensional vectors.
#   Common embedding sizes across providers:
#   - OpenAI text-embedding-3-large: 3072 dims
#   - OpenAI text-embedding-3-small: 1536 dims
#   - Google gemini-embedding-001: 3072 dims (our model)
#   - Sentence Transformers all-MiniLM-L6-v2: 384 dims
#
#   More dimensions = more semantic nuance captured, but also
#   more storage space and slightly slower search. 3072 is what
#   Google's latest embedding model produces.
#
# NOTE: If you switch embedding models, you MUST update this value
#   AND re-embed all your chunks. Vectors from different models
#   live in incompatible vector spaces.
EMBEDDING_DIMENSIONS = 3072

# ── Qdrant Vector Database Settings ───────────────────────
# WHY QDRANT INSTEAD OF CHROMADB?
#   We migrated from ChromaDB to Qdrant for several reasons:
#
#   1. ARCHITECTURE: ChromaDB is embedded (runs inside your Python
#      process using SQLite). Qdrant is client-server architecture.
#      This matters because in Docker (Stage 9), each service runs
#      in its own container. Qdrant naturally runs as a separate
#      container. ChromaDB doesn't have a good server story.
#
#   2. FILTERING: Qdrant has "payload indexing" — it can create
#      indexes on metadata fields (like startup name, year) for
#      fast filtered search. ChromaDB's filtering is basic.
#
#   3. PRODUCTION-GRADE: Companies like Microsoft, Disney, and
#      JetBrains use Qdrant in production. ChromaDB is mostly
#      used for prototyping and tutorials.
#
#   4. LOCAL MODE: Despite being a server DB, Qdrant supports
#      `QdrantClient(path="./qdrant_db")` which saves to disk
#      locally — no Docker needed for development. Best of both worlds.
#
# WHAT IS A "COLLECTION"?
#   - Same concept as ChromaDB: like a table in SQL, but for vectors
#   - Each collection stores vectors + payload (metadata + text)
#   - You can have multiple collections (e.g., "indian_startups",
#     "research_papers") in one Qdrant instance
#
# QDRANT_PATH vs QDRANT_URL:
#   - QDRANT_PATH="./qdrant_db" → local mode (for development)
#   - QDRANT_URL="http://qdrant:6333" → server mode (for Docker/production)
#   - We'll switch to URL mode in Stage 9 when we containerize
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "indian_startups")

# ── Chunking Settings ──────────────────────────────────────
# CHUNK_SIZE: How many tokens per chunk
#
# WHY 400 TOKENS (not 1000)?
# - Smaller chunks = more precise retrieval
# - If you ask "Who founded Zerodha?", a 400-token chunk about
#   Zerodha's founding is better than a 1000-token chunk that also
#   talks about their revenue, team, and office
# - Smaller chunks = less noise in the LLM's context window
# - Tradeoff: too small (100 tokens) and you lose context
#
# CHUNK_OVERLAP: How many tokens overlap between consecutive chunks
#
# WHY 50 TOKEN OVERLAP?
# - Prevents cutting sentences mid-thought at chunk boundaries
# - Example: If a sentence about "Zerodha was founded in 2010 by
#   Nithin Kamath" falls right at the boundary, overlap ensures
#   BOTH chunks contain the complete sentence
# - Tradeoff: more overlap = more storage, but safer boundaries
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Retrieval Settings ─────────────────────────────────────
# TOP_K_RETRIEVAL: How many chunks to retrieve from vector DB
#
# WHY 10?
# - We retrieve 10, then rerank to keep the best 5 (in Stage 5)
# - Retrieving too few = might miss relevant chunks
# - Retrieving too many = more noise, slower reranking
# - 10-20 is the sweet spot most companies use
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 10))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))

# ── Validation ─────────────────────────────────────────────
# Fail fast if the API key is missing. Better to crash at startup
# than to get a cryptic error 5 minutes into processing.
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found! "
        "Create a .env file with your API key. "
        "See .env.example for the template."
    )

# ── Create directories if they don't exist ─────────────────
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
