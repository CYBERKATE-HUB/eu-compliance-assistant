# GDPR + AI Act Compliance Assistant — Architecture

## What this system does

A bilingual RAG assistant that answers questions about the GDPR and EU AI Act.
The user asks a question in English or French. The system retrieves the most
relevant articles from the official regulation texts, passes them to Mistral
as grounded context, and returns an answer with specific article citations.

It does not use general LLM knowledge — every claim traces back to a retrieved
article. This is a deliberate design choice to prevent hallucination in a
legal context.

---

## Project structure
gdpr-assistant/
├── app/
│   ├── core/
│   │   ├── parse_documents.py  # HTML → article chunks
│   │   ├── embed_store.py      # chunks → vectors → ChromaDB
│   │   └── retrieval.py        # hybrid search + direct article lookup
│   ├── assistant.py            # CLI interface
│   └── explain.py              # educational explainer (shows internals)
├── data/
│   ├── all_chunks.json         # 424 parsed article chunks
│   └── gdpr_db/                # ChromaDB vectors (not in git, rebuilds)
├── .env.example                # API key template
├── ARCHITECTURE.md
└── requirements.txt

---

## Source documents

| Document | Language | Articles |
|----------|----------|---------|
| GDPR (2016/679) | English | 99 |
| GDPR (2016/679) | French | 95 |
| AI Act (2024/1689) | English | 113 |
| AI Act (2024/1689) | French | 117 |
| **Total** | | **424 chunks** |

Source: EUR-Lex official HTML versions. HTML chosen over PDF because
it has semantic structure (headings, article tags) that makes
article-boundary parsing reliable.

---

## Chunking strategy

Each article is one chunk. Metadata stored per chunk:
`article_number`, `article_title`, `source` (gdpr/ai_act), `language` (en/fr).

**Why article-boundary chunking:**
GDPR and AI Act have natural legal divisions — each article is a
complete, self-contained unit of law. Splitting mid-article breaks
legal conditions and context. Sliding windows would duplicate text
across chunks and dilute retrieval precision.

---

## Retrieval pipeline

### Stage 1 — Hybrid search (dense + sparse)

Two parallel search methods run on every query:

**Dense search (ChromaDB + HNSW):**
- Query is embedded with `all-MiniLM-L6-v2` (384 dimensions)
- ChromaDB finds nearest vectors by cosine similarity
- Good for semantic queries: "can I ask a company to forget me" → finds Article 17

**Sparse search (BM25):**
- Keyword frequency scoring across raw article text
- Good for exact legal strings: "Article 17", "data controller"
- Dense search alone misses these because embeddings treat them as generic tokens

**Why all-MiniLM-L6-v2:**
Fast, free, 384 dimensions, handles English/French mixed text well.
Good quality for a portfolio demo without requiring GPU inference.

**Why ChromaDB:**
Runs locally with zero infrastructure. Pure Python. Sufficient for
424 chunks. For production at scale: Qdrant self-hosted.

### Stage 2 — Reciprocal Rank Fusion (RRF)

Dense and sparse results are merged by rank position, not raw score.
This avoids the scaling problem of comparing cosine distances (0–1)
with BM25 frequencies (0–∞).
score(doc) = Σ 1 / (k + rank)    k = 60

k=60 is the standard default — it dampens the effect of rank-1,
reducing the impact of outliers. A document appearing in both lists
gets two contributions and naturally ranks higher.

### Direct article lookup

If the query contains "Article X", the system bypasses semantic search
and fetches that article directly via `get_article_by_number()`.
This solves a known failure mode where BM25 treats "article" and "17"
as separate high-frequency tokens across all 424 chunks.

### Language detection

French queries are detected by checking for common French function words
(les, des, est, sont, pour, dans…). The system then filters retrieval
to French-language chunks only, keeping IDF scores clean.

---

## Hallucination mitigation — three layers

1. **Confidence threshold:** top retrieval cosine score checked before
   calling Mistral. If below 0.4, return a fallback message — never
   generate with weak context.

2. **Grounding prompt:** system prompt explicitly forbids outside
   knowledge: "Answer ONLY using the provided context articles."

3. **Citation requirement:** every factual claim must reference
   [Article X]. If the model cannot cite it, it should not say it.

---

## Known failure modes

**Article number queries are imprecise without direct lookup:**
BM25 treats "Article" and "17" as separate tokens appearing in all
424 chunks, diluting the score. Solved by `get_article_by_number()`
for explicit article references.

**Long articles lose granularity:**
Article 4 (GDPR Definitions) contains 26 definitions in one chunk.
Sub-article chunking would improve precision for definition queries.
Not implemented in v1.

**No conversation memory:**
Each question is independent — no message history passed to Mistral.
Next iteration: add conversation history to the API call.

**Reranker not implemented:**
A cross-encoder reranker (BAAI/bge-reranker-base) would re-score
candidates by reading query + document together. More accurate than
embedding distance alone, ~150ms latency cost. Kept out of v1 to
stay focused on core architecture.

---

## What would change for production at 10,000 users

| Component | Current (demo) | Production |
|-----------|---------------|------------|
| Vector DB | ChromaDB local | Qdrant self-hosted (EU region) |
| LLM | mistral-large-latest | mistral-large with rate limiting + caching |
| Reranker | not implemented | bge-reranker-base, async batching |
| Language detection | keyword list | fasttext classifier |
| API | CLI only | FastAPI with auth + rate limits |
| Evaluation | manual testing | RAGAS: faithfulness + context recall |
| Hosting | local | EU-hosted (GDPR compliance for the tool itself) |

---

## How to run

```bash
# 1. Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your Mistral API key

# 2. Parse documents (run once)
python app/core/parse_documents.py

# 3. Build vector index (run once)
python app/core/embed_store.py

# 4. Ask questions
python -m app.assistant

# 5. See internals
python app/explain.py
```

---