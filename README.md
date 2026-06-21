# EU Compliance Platform

A bilingual RAG assistant and support agent for GDPR and EU AI Act compliance.
Built with Mistral, ChromaDB, and hybrid search. Answers in English and French.

## Two tools, one retrieval layer

**RAG Compliance Assistant** — ask any question about GDPR or EU AI Act, get a grounded answer with specific article citations. No hallucination — every claim traces back to a retrieved article.

**GDPR Support Agent** — paste a support ticket, get a classified, legally-grounded DRAFT response ready for human review. Uses a ReAct loop with three tools: classify, search, memory.

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding | all-MiniLM-L6-v2 | Fast, free, handles EN/FR |
| Vector DB | ChromaDB | Local, zero infra, pure Python |
| Sparse search | BM25 | Catches exact article references |
| Merge | RRF k=60 | Scale-invariant rank fusion |
| LLM | Mistral large | EU-hosted, strong on legal output |
| Agent pattern | ReAct | Model decides tool sequence |

## Sources

Download from EUR-Lex and save to project root before running:

| File | URL |
|------|-----|
| `gdpr.html` | EUR-Lex → 32016R0679 → EN → HTML |
| `gdpr_fr.html` | EUR-Lex → 32016R0679 → FR → HTML |
| `ai_act_en.html` | EUR-Lex → 32024R1689 → EN → HTML |
| `ai_act_fr.html` | EUR-Lex → 32024R1689 → FR → HTML |

424 article chunks total across 4 documents.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your Mistral API key

# Build knowledge base (run once)
python app/core/parse_documents.py
python app/core/embed_store.py

# Build agent memory (run once)
python -m support_agent.seed_memory
```

## Run

```bash
# RAG Assistant
python -m app.assistant        # ask questions
python app/explain.py          # see retrieval internals

# Support Agent
python -m support_agent.agent          # process tickets
python -m support_agent.agent_explain  # see agent internals
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full technical documentation including:
- Chunking strategy and why article-boundary splitting
- Hybrid search: dense vs sparse, why both are needed
- RRF merge formula and k=60 rationale
- ReAct agent loop and tool design decisions
- Hallucination mitigation: three-layer defence
- Known failure modes and production roadmap
- Interview walkthrough answer