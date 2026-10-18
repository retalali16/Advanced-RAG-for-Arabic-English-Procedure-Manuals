# Advanced RAG for Arabic/English Procedure Manuals

An end-to-end Retrieval-Augmented Generation system that answers questions over three bilingual (Arabic/English) operational manuals. A **query router** decides how much retrieval effort each question needs, runs only the selected techniques, and every answer is scored for faithfulness, relevance, correctness, cost, and latency.

**Corpus:** Assets & Warehouse Operations, Central Alarm, and Central Mail & Files Unit procedure manuals.

---

## Features

- **Smart routing:** each question goes to `simple`, `basic_rag`, or `advanced_rag`, returned as JSON `{route, reason, techniques}`.
- **Arabic-first ingestion:** pages are rendered and OCR'd with Tesseract (`ara+eng`) to avoid a font-cmap glyph bug in `pypdf` that silently swaps Arabic letters. Falls back to `pypdf` when OCR is unavailable or scores worse on a page.
- **Structure-aware chunking:** splits on paragraph and numbered-step boundaries (1100 chars, 180 overlap), so procedure steps are never cut in half. Each chunk keeps source, page, year, and extraction-method metadata for filtering and citations.
- **Hybrid retrieval:** BM25 sparse search fused with dense **BGE-M3** embeddings (multilingual, runs locally on CPU, no API key).
- **Advanced techniques, applied only when needed:**
  - Query rewriting (resolves follow-ups like "why is it needed?")
  - Multi-query with Reciprocal Rank Fusion (RRF)
  - Question decomposition (comparisons, multi-part questions)
  - HyDE (hypothetical document embeddings)
  - Self-query (metadata filters such as year or source manual)
  - CRAG (corrective re-search when retrieved context is weak)
  - Keyword/LLM reranking and contextual compression
- **Bilingual glossary:** English questions retrieve the right Arabic sections.
- **Evaluation harness:** Context Relevance, Faithfulness, Answer Relevance, Correctness, cost, and latency per question, with a forced basic-RAG baseline for fair comparison.
- **Multiple LLM backends:** OpenRouter, Claude, Ollama, or Gemini, plus a fully offline extractive fallback when no key is set.
- **Streamlit chat UI** and a CLI for ad-hoc questions.

---

## Architecture

```
Question
   │
   ▼
Router ──► simple ──────────────────────────────► Answer
   │
   ├──► basic_rag ──► Hybrid search ─────────────► Grounded generation ──► Evaluation
   │
   └──► advanced_rag
          │
          ▼
   Rewrite / Multi-query / Decomposition / HyDE / Self-query
          │
          ▼
   Hybrid search (BM25 + BGE-M3) ──► RRF merge ──► CRAG retry ──► Rerank ──► Compression
          │
          ▼
   Grounded generation (with citations) ──► Evaluation
```

---

## Project Structure

```
.
├── advanced_rag/
│   ├── ingest.py        # OCR + structure-aware chunking
│   ├── store.py         # BM25 + dense hybrid vector store
│   ├── embeddings_bge.py# Local BGE-M3 embeddings
│   ├── router.py        # simple / basic_rag / advanced_rag routing
│   ├── transforms.py    # rewrite, multi-query, decompose, HyDE, self-query
│   ├── retrieve.py      # RRF, rerank, compression, CRAG
│   ├── generate.py      # grounded answer generation
│   ├── evaluate.py      # scoring (LLM judge + heuristic fallback)
│   ├── bilingual.py     # Arabic/English glossary
│   ├── cost.py          # token and cost tracking
│   ├── llm.py           # multi-backend LLM client
│   └── pipeline.py      # RAGPipeline orchestration
├── streamlit_app.py     # Chat UI
├── ask.py               # CLI for ad-hoc questions
├── run_eval.py          # Full evaluation run
├── data/corpus/         # Source PDFs
├── outputs/             # Results tables and JSON
└── reports/REPORT.md    # Analysis report
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
cp .env.example .env     # Windows: copy .env.example .env
```

### 2. Install OCR tools (needed for good Arabic quality)

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr tesseract-ocr-ara poppler-utils
```

On Windows, install [Tesseract (with the Arabic language pack)](https://github.com/UB-Mannheim/tesseract/wiki) and [poppler](https://github.com/oschwartz10612/poppler-windows/releases), and add both to `PATH`. Without them, ingestion falls back to `pypdf` (lower Arabic quality). Set `USE_OCR=0` to force this.

### 3. Configure `.env`

Set one LLM backend (or leave all unset for offline extractive mode):

| Variable | Purpose |
|---|---|
| `USE_OPENROUTER`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | OpenRouter backend |
| `USE_CLAUDE`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` | Claude backend |
| `USE_OLLAMA`, `OLLAMA_BASE_URL`, `OLLAMA_GENERATION_MODEL` | Local Ollama backend |
| `GEMINI_API_KEY` | Gemini backend |
| `EMBEDDING_BACKEND` | `bge_m3` (default), `gemini`, or `none` |
| `TOP_K`, `MAX_CONTEXT_CHARS`, `GENERATION_MAX_TOKENS` | Retrieval and generation limits |

> **Never commit a real API key.** `.env` is gitignored; keep only placeholders in `.env.example`.

---

## Usage

**Chat UI**

```bash
streamlit run streamlit_app.py
```

**Ask a single question (CLI)**

```bash
python ask.py "How does the mail unit handle a lost registered item?"
python ask.py --basic "..."                       # force the basic_rag baseline
python ask.py --file arabic_test_questions.txt    # batch mode
```

Results are also saved as UTF-8 text under `outputs/`, because some terminals render Arabic text reversed.

**Run the full evaluation**

```bash
python run_eval.py
```

This re-ingests the PDFs, re-embeds all chunks, runs the 10-question evaluation set through both the routed pipeline and the basic-RAG baseline, and writes:

- `outputs/RESULTS_TABLE.md`: results tables
- `outputs/evaluation.json`: answers, retrieved chunks, scores
- `outputs/route_decisions.json`: router decisions
- `outputs/cost_breakdown.json`: per-step cost
- `reports/REPORT.md`: analysis

---

## Embeddings

Dense retrieval uses **BAAI/bge-m3** loaded locally through `sentence-transformers`:

- Multilingual and Arabic-capable
- ~2.3 GB, 568M parameters, runs on CPU for this corpus size
- No API key, no quota, works offline once cached
- Symmetric encoder: no instruction prefix needed for queries or documents

If `sentence-transformers` or `torch` is missing, the store logs a warning and falls back to BM25-only retrieval instead of crashing.

---

## Evaluation Notes

Each question is scored on Context Relevance, Faithfulness, Answer Relevance, and Correctness, along with cost and latency. Reference answers in `advanced_rag/questions.py` were verified against the OCR'd corpus and carry page citations.

- Context Relevance on the `simple` route is reported as `N/A` and excluded from averages, since no retrieval runs there.
- Without an LLM key, scoring uses a heuristic judge. For meaningful scores, run with an LLM judge configured.

---

## Tech Stack

Python · Streamlit · BM25 (`rank_bm25`) · BGE-M3 (`sentence-transformers`, PyTorch) · scikit-learn · Tesseract OCR · pypdf · OpenRouter / Claude / Ollama / Gemini

---

## License

Add a license of your choice (e.g. MIT) before publishing.
