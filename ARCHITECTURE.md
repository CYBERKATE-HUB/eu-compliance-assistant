# EU Compliance Platform — Architecture

## What this platform does

Two connected tools for EU regulatory compliance:

**1. RAG Compliance Assistant** — answers questions about GDPR and EU AI Act in English and French. Retrieves relevant articles from official EUR-Lex texts, passes them to Mistral as grounded context, returns answers with specific article citations.

**2. GDPR Support Agent** — takes a support ticket, classifies the issue, searches the knowledge base, retrieves similar past resolved tickets, and drafts a response for human review.

Both tools share the same retrieval layer. No general LLM knowledge is used — every claim traces back to a retrieved article.

---

## Project structure
eu-compliance-assistant/

├── app/

│   ├── core/

│   │   ├── parse_documents.py  # HTML → article chunks

│   │   ├── embed_store.py      # chunks → vectors → ChromaDB

│   │   └── retrieval.py        # hybrid search + direct article lookup

│   ├── assistant.py            # RAG assistant CLI

│   └── explain.py              # educational explainer

├── support_agent/

│   ├── tools.py                # three agent tools

│   ├── agent.py                # ReAct loop + CLI

│   ├── agent_explain.py        # agent educational explainer

│   ├── seed_tickets.py         # sample resolved tickets

│   └── seed_memory.py          # indexes tickets into ChromaDB

├── data/

│   └── gdpr_db/                # ChromaDB vectors (not in git, rebuilds)

├── .env.example

├── ARCHITECTURE.md

└── requirements.txt

---

## Source documents

| Document | Language | Articles |
|----------|----------|----------|
| GDPR (2016/679) | English | 99 |
| GDPR (2016/679) | French | 95 |
| AI Act (2024/1689) | English | 113 |
| AI Act (2024/1689) | French | 117 |
| **Total** | | **424 chunks** |

Source: EUR-Lex official HTML versions. HTML chosen over PDF because it has semantic structure that makes article-boundary parsing reliable.

---

## Shared retrieval layer

Both tools use the same retrieval pipeline from `app/core/retrieval.py`.

### Chunking strategy

Each article is one chunk. Metadata per chunk: `article_number`, `article_title`, `source` (gdpr/ai_act), `language` (en/fr).

**Why article-boundary chunking:** GDPR and AI Act have natural legal divisions — each article is a complete, self-contained unit of law. Splitting mid-article breaks legal conditions. Sliding windows duplicate content and dilute retrieval precision.

### Hybrid search (dense + sparse)

**Dense search (ChromaDB + HNSW):**
- Query embedded with `all-MiniLM-L6-v2` (384 dimensions)
- Finds nearest vectors by cosine similarity
- Good for semantic queries: "can I ask a company to forget me" → Article 17

**Sparse search (BM25):**
- Keyword frequency scoring across raw text
- Good for exact legal strings: "Article 17", "data controller"
- Dense search alone misses these — embeddings treat them as generic tokens

**Why all-MiniLM-L6-v2:** Fast, free, 384 dimensions, handles EN/FR mixed text. No GPU required.

**Why ChromaDB:** Runs locally, zero infrastructure, pure Python. For production: Qdrant self-hosted.

### Reciprocal Rank Fusion (RRF)

Dense and sparse results merged by rank position, not raw score. Avoids the scaling problem of comparing cosine distances (0–1) with BM25 frequencies (0–∞).
score(doc) = Σ 1 / (k + rank)    k = 60

k=60 dampens the effect of rank-1, reducing outlier impact. A document appearing in both lists gets two contributions and ranks higher.

### Direct article lookup

If query contains "Article X", system bypasses semantic search and fetches that article directly. Solves the failure mode where BM25 treats "article" and "17" as separate high-frequency tokens across all 424 chunks.

### Language detection

French queries detected by checking common French function words. System filters retrieval to French-language chunks only, keeping IDF scores clean.

---

## RAG Compliance Assistant

### Hallucination mitigation — three layers

1. **Confidence threshold:** top retrieval cosine score checked before calling Mistral. Below 0.4 → return fallback, never generate with weak context.
2. **Grounding prompt:** "Answer ONLY using the provided context articles."
3. **Citation requirement:** every factual claim must reference [Article X].

### Known failure modes

**Long articles lose granularity:** Article 4 (GDPR Definitions) has 26 definitions in one chunk. Sub-article chunking would improve precision. Not in v1.

**No conversation memory:** each question is independent. Next iteration: add message history to the API call.

**Reranker not implemented:** cross-encoder (BAAI/bge-reranker-base) would re-score candidates by reading query + document together. ~150ms latency cost. Kept out of v1.

---

## GDPR Support Agent

### Architecture pattern: ReAct

Unlike the RAG assistant (fixed pipeline), the agent uses ReAct — Reason + Act:
User ticket

↓

Mistral reasons → calls classify_ticket

↓

Mistral sees result → calls search_gdpr_kb

↓

Mistral sees articles → calls get_similar_resolved_tickets (optional)

↓

Mistral decides: enough context → writes DRAFT

↓

Human review

Mistral decides the tool sequence — not hardcoded steps.

### Three tools

| Tool | Does | Why separate |
|------|------|-------------|
| `classify_ticket` | Categorises issue type and urgency via Mistral JSON output | Wrong category = wrong search query = wrong articles |
| `search_gdpr_kb` | Retrieves relevant articles — reuses shared retrieval layer | Separation of concerns: classification ≠ retrieval |
| `get_similar_resolved_tickets` | Finds past resolved tickets by semantic similarity | Memory improves draft quality with proven solutions |

### Why classify before searching?

A vague query "employee wants data deleted" returns Article 70 (Tasks of the Board). A classified query with subcategory `right_to_erasure` maps via `GDPR_QUERY_MAP` to "right to erasure data subject request Article 17" — returns Article 17 correctly.

Classification is query enrichment, not just labelling.

### Agent memory

Past resolved tickets stored as separate ChromaDB collection (`resolved_tickets`). Same embedding model as the knowledge base — semantic similarity search across ticket descriptions. 5 seed tickets for demo. In production: auto-indexed from resolved ticket queue.

### Hallucination mitigation

Same three layers as the RAG assistant, plus:
- All output marked **DRAFT** — never sent to customer directly
- Human review gate before any response goes out

### Known failure modes

**Tool loop:** agent called `search_gdpr_kb` 4 times on ambiguous tickets. Fix: explicit instruction in system prompt — maximum 2 searches per ticket.

**Rate limiting:** Mistral API returns 429 on rapid sequential calls. Fix in demo: `time.sleep(4)`. Fix in production: exponential backoff with jitter.

---

## Production considerations

| Component | Current (demo) | Production |
|-----------|---------------|------------|
| Vector DB | ChromaDB local | Qdrant self-hosted (EU region) |
| LLM | mistral-large-latest | mistral-large with rate limiting + caching |
| Reranker | not implemented | bge-reranker-base, async batching |
| Language detection | keyword list | fasttext classifier |
| Agent rate limiting | time.sleep(4) | Exponential backoff + request queue |
| Human review | printed DRAFT | Ticketing system (Zendesk, Jira) |
| Monitoring | print statements | LLM observability (Langfuse, Helicone) |
| Evaluation | manual | RAGAS: faithfulness + context recall |
| Hosting | local | EU-hosted (GDPR compliant) |

---

## How to run

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add Mistral API key

# Build knowledge base (run once)
python app/core/parse_documents.py
python app/core/embed_store.py

# Build agent memory (run once)
python -m support_agent.seed_memory

# RAG Assistant
python -m app.assistant        # ask questions
python app/explain.py          # see retrieval internals

# Support Agent
python -m support_agent.agent          # process tickets
python -m support_agent.agent_explain  # see agent internals
```
---