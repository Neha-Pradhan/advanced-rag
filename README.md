# Advanced RAG — NCERT Science Tutor

A grade-aware Retrieval-Augmented Generation system over NCERT Grade 7 & 8 Science textbooks. Answers a student's questions using only their textbook, cites chapter and section, points to relevant figures, and honestly refuses questions outside the corpus.

Built as a hand-rolled pipeline (not a framework wrapper) to understand what production RAG systems actually do — then measured with RAGAS and tracked in MLflow.

## What it does

- **Grade-aware retrieval** — a Grade 8 student's queries search Grade 8 *and* lower content (`grade ≤ user`), so explanations use their level plus earlier references, never higher.
- **Grounded answers** — the LLM answers *only* from retrieved textbook passages, cites `Chapter X, Section Y`, and points to figures ("see Fig. 2.5").
- **Honest "not found" handling** — a two-stage gate (similarity floor + LLM relevance check) rejects out-of-corpus questions instead of hallucinating an answer.
- **Streamlit UI** — a simple study interface with example questions.

## Architecture

Two pipelines that never run at the same time — an offline ingestion pipeline (batch, runs once) and an online query pipeline (per request). This mirrors production: a batch job vs an API service.

**Ingestion (offline):**
```
NCERT PDFs → extract (clean text) → chunk (section-aware, 1074 chunks)
→ embed (MiniLM, 384-dim) → Qdrant collection
```

**Query (online):**
```
question + grade → grade-filtered vector search → two-stage gate
→ grounded answer (cited, figure refs)  OR  consent fallback
```

An evaluation harness (RAGAS + MLflow) wraps the query pipeline so each retrieval technique can be measured before/after.

## Key design decisions

- **Section-aware chunking** — NCERT chapters have explicit `N.N` section headings, so chunks are cut on the document's own structure rather than fixed character windows. Oversized sections are sub-split on sentence boundaries with the heading re-injected into each sub-chunk (so the embedding always captures the section topic). Recursive splitting would be the right default only for unstructured corpora.
- **Single Qdrant collection, grade in payload** — the core query (`grade ≤ N`) spans grades, so one filtered search beats sharding by grade. Payload filtering requires a payload index (Qdrant refuses to filter an unindexed field).
- **Two-stage not-found gate** — vector search *always* returns something (nearest ≠ relevant). A cheap similarity floor rejects obvious misses; an LLM relevance check catches the harder "high score but irrelevant" cases before answering.
- **Metadata is not embedded** — only chunk text becomes a vector; grade/chapter/section/figures ride alongside as payload for filtering and citation. The one exception is the section heading, which lives in both (embedded *and* stored) because each copy serves a different consumer.
- **Model-agnostic LLM layer** — generation and the relevance gate sit behind one function, so the provider is swappable.

## Stack

| Layer | Choice |
|---|---|
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Vector store | Qdrant Cloud (payload-filtered search) |
| Extraction / chunking | pypdf + hand-rolled section splitter, LangChain `RecursiveCharacterTextSplitter` for sub-splitting |
| Generation & gate | Gemini (free tier) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| Experiment tracking | MLflow |
| UI | Streamlit |

## Evaluation (baseline)

Measured with RAGAS over a hand-built eval set (in-corpus + out-of-corpus questions):

| Metric | Score |
|---|---|
| Faithfulness | 0.94 |
| Answer relevancy | 0.91 |
| Context precision | 0.86 |
| Context recall | 0.69 |

Read: generation is strong (grounding constraint works, low hallucination); **retrieval recall is the bottleneck** (~31% of needed context missed) — the clear target for the next round of improvements.

## What I learned building this

- Chunking bugs are invisible in code and only surface by inspecting real output — a chapter with no `N.N` headings silently produced zero chunks; another repeated its section heading as a per-page running header and over-fragmented into false sections.
- Extraction is corpus-specific judgment, not a library call — the cleaning rules (running headers, worksheet dots, non-breaking spaces) are unique to these PDFs.
- Vector search always returns something, so a not-found gate is essential — the naive systems that skip it are exactly the ones that confidently answer out-of-corpus questions.
- Real users break assumptions eval sets don't — the first out-of-scope question a real student asked exposed a fallback message that promised content the system didn't have.

## Roadmap

- **Figure display** — extract diagram images from the PDFs and show them alongside answers.
- **Hybrid search** — add sparse/keyword retrieval to close the recall gap on exact-term queries.
- **Multi-source corpus** — blend NCERT with a teacher's supplementary notes (incl. OCR of handwritten material).
- **MCP server** — expose the retrieval core as a second transport alongside the Streamlit UI.
- **Graph RAG** — entity/relationship graph for multi-fact and cross-chapter questions.

## Running it

```bash
# in the `agents` conda env, with QDRANT_URL / QDRANT_API_KEY / GEMINI_API_KEY in .env
pip install -r requirements.txt

# ingest (one time)
python -m src.extract
python -m src.chunk
python -m src.embed

# launch the tutor
streamlit run app/study_tool.py
```

Corpus PDFs are not included (NCERT copyright) — download Grade 7 & 8 Science from ncert.nic.in into `data/raw/grade7/` and `data/raw/grade8/`.
