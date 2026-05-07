# 🧠 Brief AI — Advanced Corrective RAG System

> An advanced Retrieval-Augmented Generation pipeline for the Indian startup ecosystem, implementing HyDE, multi-query retrieval, cross-encoder reranking, and self-correction.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)

## 🏗️ Architecture

```
User Query → Multi-Query + HyDE → ChromaDB Retrieval → BGE Reranking → Gemini Generation → Self-Correction → Answer + Citations
```

## ✨ Features

- **Multi-Query Retrieval** — Generates 3 query rephrasings to capture different intent facets
- **HyDE (Hypothetical Document Embeddings)** — Embeds a hypothetical answer for better retrieval
- **Cross-Encoder Reranking** — BGE reranker rescores candidates for precision
- **Self-Correction** — LLM evaluates its own answer against source chunks
- **Citations** — Every claim is backed by source references
- **Confidence Scoring** — Transparent reliability metrics

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Chaplot05/Brief-AI.git
cd Brief-AI

# Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# Ingest data
python scripts/ingest.py

# Query
python -c "from src.pipeline.rag_pipeline import query_rag; print(query_rag('What are the top Indian unicorns?')['answer'])"
```

## 📁 Project Structure

```
src/
├── config.py              # Centralized configuration
├── ingestion/             # Data pipeline (scrape → chunk → embed)
├── retrieval/             # Smart retrieval (basic, multi-query, HyDE)
├── generation/            # LLM generation + self-correction
├── pipeline/              # Orchestration
├── evaluation/            # RAGAS-based evaluation
└── api/                   # FastAPI backend
```

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embeddings | Gemini Embedding 001 | Free, high-quality 768-dim vectors |
| Vector DB | ChromaDB | Local, zero-cost, Python-native |
| LLM | Gemini 2.0 Flash | Fast, free tier, good instruction following |
| Reranker | BGE Reranker v2 | SOTA open-source cross-encoder |
| Framework | LangChain | Industry-standard LLM orchestration |
| API | FastAPI | Async, auto-docs, type-safe |
| Frontend | Streamlit | Rapid AI demo prototyping |

## 📄 License

MIT
