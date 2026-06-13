# explain.py
"""
Sovereign EU Compliance Assistant - Technical Telemetry Engine
Implements a Two-Stage Hybrid Retrieval-Reranking Pipeline (Recall -> Precision).
Designed by the AI Deployment Strategist to showcase search-space telemetry.
"""

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

# =====================================================================
# STAGE 0: STATE ENCAPSULATION & DATA RESTORATION
# =====================================================================

print("🔄 Bootstrapping local telemetry tracking instances...")
try:
    with open("all_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "Critical Error: 'all_chunks.json' missing from execution root path. "
        "Please ensure 'parse_documents.py' and 'embed_store.py' have been executed."
    )

# Extract and isolate distinct language streams to protect Inverse Document Frequency (IDF) limits
chunks_en = [c for c in chunks if c.get("language") == "en"]
chunks_fr = [c for c in chunks if c.get("language") == "fr"]

# Level 1 Vector Models (Optimized for High Recall & O(1) Search Latency)
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./gdpr_db")
collection = chroma_client.get_or_create_collection(
    name="gdpr_articles",
    metadata={"hnsw:space": "cosine"}
)

# Level 1 Tokenized Sparse Inversion In-Memory Mapping
bm25_en = BM25Okapi([c["text"].lower().split() for c in chunks_en])
bm25_fr = BM25Okapi([c["text"].lower().split() for c in chunks_fr])

# Level 2 Transformer Models (Optimized for Context Precision & Joint Feature Matrix Validation)
reranker = CrossEncoder("BAAI/bge-reranker-base")


# =====================================================================
# CORE ALGORITHMIC CORE FUNCTIONS
# =====================================================================

def detect_language(query: str) -> str:
    """Deterministic language router preventing vocabulary drift."""
    french_indicators = {"quel", "quels", "quelle", "quelles", "est", "sont", "les", "des", "une", "pour", "dans"}
    tokens = set(query.lower().split())
    return "fr" if tokens.intersection(french_indicators) else "en"

def calculate_rrf(dense_ids: list, sparse_ids: list, k: int = 60) -> list:
    """
    Computes reciprocal rank scoring matrix to normalize independent retrieval layers.
    Protects downstream nodes from raw model output value anomalies.
    """
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda node: node[1], reverse=True)

def render_section(title: str):
    print(f"\n{'#' * 75}\n# 🔍 {title.upper()}\n{'#' * 75}")


# =====================================================================
# THE TELEMETRY PIPELINE EXECUTION LOOP
# =====================================================================

def execute_telemetry_analysis(query: str):
    render_section("Starting Pipeline Telemetry Analysis")
    lang = detect_language(query)
    active_chunks = chunks_en if lang == "en" else chunks_fr
    active_bm25 = bm25_en if lang == "en" else bm25_fr
    
    print(f"📥 Input Query : \"{query}\"")
    print(f"🌐 Target Path : Localized {lang.upper()} Legal Repository Stream Only")

    # ────────────────────────────────────────────────────────
    # STAGE 1: SYSTEM INDEX BOUNDARIES
    # ────────────────────────────────────────────────────────
    render_section("Stage 1 — Chunk Ingestion Design")
    print("""  [Architecture Pattern]: Deterministic Article-Boundary Segmentation.
  [Strategy Rationale]  : Legal documents form interdependent semantic structures. 
                          Arbitrary character limits risk splitting definitions, 
                          while sliding windows dilute retrieval clarity. Forcing 
                          1 Chunk = 1 Native Regulation Article prevents cross-context pollution.""")
    sample_chunk = active_chunks[0]
    print(f"\n  Representative Target Object Structure Layout:")
    print(f"  • Legal Source  : {sample_chunk['source'].upper()}")
    print(f"  • Reference ID  : {sample_chunk['article_number']} ({sample_chunk['article_title']})")
    print(f"  • Ingest Volume : {len(sample_chunk['text'])} characters")

    # ────────────────────────────────────────────────────────
    # STAGE 2: DENSE EMBEDDING EVALUATION
    # ────────────────────────────────────────────────────────
    render_section("Stage 2 — Dense Semantic HNSW Processing")
    query_vector = bi_encoder.encode(query)
    
    dense_res = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=10,
        where={"language": lang},
        include=["metadatas", "distances"]
    )
    
    dense_hits = []
    dense_ids = []
    if dense_res and dense_res["ids"] and len(dense_res["ids"][0]) > 0:
        dense_ids = dense_res["ids"][0]
        for idx in range(len(dense_ids)):
            meta = dense_res["metadatas"][0][idx]
            dense_hits.append({
                "id": dense_ids[idx],
                "article": meta["article_number"],
                "score": round(1 - dense_res["distances"][0][idx], 4)
            })

    print("  Vector Model Architecture: 384 Float Dimensions (all-MiniLM-L6-v2)")
    print(f"  Raw Vector Preview (First 8 Index Dimensions): {query_vector[:8].tolist()}")
    print("\n  Top 5 ChromaDB Retrieval Hits (HNSW Cosine Vector Matches):")
    for hit in dense_hits[:5]:
        print(f"   ↳ ID: {hit['id']:<12} | {hit['article']:<12} | Cosine Proximity Vector: {hit['score']:+.4f}")

    # ────────────────────────────────────────────────────────
    # STAGE 3: SPARSE TERM FREQUENCY EXTRACTION
    # ────────────────────────────────────────────────────────
    render_section("Stage 3 — Sparse Term-Frequency Indexing (BM25)")
    tokens = query.lower().split()
    bm25_scores = active_bm25.get_scores(tokens)
    top_sparse_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    sparse_ids = [f"article_{i}" for i in top_sparse_idx]
    
    sparse_hits = [
        {"id": f"article_{idx}", "article": active_chunks[idx]["article_number"], "score": round(float(bm25_scores[idx]), 4)}
        for idx in top_sparse_idx
    ]

    print(f"  Tokenized Search Expressions: {tokens}")
    print("\n  Top 5 BM25 Retrieval Hits (Term Frequency Inverse Document Frequency):")
    for hit in sparse_hits[:5]:
        print(f"   ↳ ID: {hit['id']:<12} | {hit['article']:<12} | Score Weight: {hit['score']:+.4f}")

    # ────────────────────────────────────────────────────────
    # STAGE 4: RECIPROCAL RANK FUSION ORCHESTRATION
    # ────────────────────────────────────────────────────────
    render_section("Stage 4 — Interleaved Rank Normalization (RRF)")
    rrf_fused = calculate_rrf(dense_ids, sparse_ids, k=60)
    
    print("  [Mathematical Logic]: Merges uncalibrated Dense percentages with Sparse scale variants.")
    print("  [Mathematical Formula]: Target Weight = SUM( 1 / (60 + Stage_1_Ordinal_Position) )")
    print("\n  Top 5 Fused Hybrid Candidate Pool Profiles:")
    
    candidate_chunks_pool = []
    for rank, (doc_id, rrf_score) in enumerate(rrf_fused[:10]):
        if doc_id.startswith("article_"):
            target_idx = int(doc_id.split("_")[1])
            chunk_data = active_chunks[target_idx]
        else:
            chunk_data = next((c for c in chunks if c["article_number"] in doc_id and c["language"] == lang), active_chunks[0])
            
        candidate_chunks_pool.append(chunk_data)
        if rank < 5:
            print(f"   ↳ Rank {rank+1}: {chunk_data['article_number']:<12} | Unified Hybrid RRF Weight: {rrf_score:.5f}")

    # ────────────────────────────────────────────────────────
    # STAGE 5: CROSS-ENCODER ATTENTION RE-CALCULATION
    # ────────────────────────────────────────────────────────
    render_section("Stage 5 — Stage 2 Deep Precision Reranking (Cross-Encoder)")
    print("""  Why inject a secondary Cross-Encoder attention tier?
  - Bi-Encoders map queries and documents separately, which can overlook deep contextual links.
  - Cross-Encoders process BOTH text strings simultaneously at query-time.
  - This step surfaces fine-grained alignment details that simple spatial closeness can miss.""")

    rerank_pairs = [[query, chunk["text"]] for chunk in candidate_chunks_pool]
    cross_scores = reranker.predict(rerank_pairs)
    
    reranked_telemetry = []
    for idx, score in enumerate(cross_scores):
        reranked_telemetry.append({
            "chunk": candidate_chunks_pool[idx],
            "initial_rrf_rank": idx + 1,
            "score": float(score)
        })
        
    reranked_telemetry = sorted(reranked_telemetry, key=lambda x: x["score"], reverse=True)

    print("\n  [Cross-Encoder Rank Shift Execution Metrics Matrix Table]")
    print(f"  {'Article Number':<16} | {'Stage 1 Rank (RRF)':<22} | {'Stage 2 Attention Score':<24}")
    print(f"  {'-'*16}-+-{'-'*22}-+-{'-'*24}")
    for final_rank, item in enumerate(reranked_telemetry[:5]):
        print(f"  {item['chunk']['article_number']:<16} | Rank {item['initial_rrf_rank']:<17} | {item['score']:+.4f} (Final Position: {final_rank+1})")

    # ────────────────────────────────────────────────────────
    # STAGE 6: GROUNDED SYNTHESIS DELIVERABLES
    # ────────────────────────────────────────────────────────
    render_section("Stage 6 — Context Synthesis Payload Configuration")
    final_context_nodes = reranked_telemetry[:3]
    print(f"  Final Target Nodes Package passed to Sovereign LLM Engine Context: {len(final_context_nodes)}")
    for i, item in enumerate(final_context_nodes):
         print(f"   📥 Node [{i+1}] -> Reference ID: {item['chunk']['article_number']} | Character Payload Volume: {len(item['chunk']['text'])}")
         
    print("""\n  Active System Guardrails:
   1. Strict contextual isolation system prompt (disallows reliance on external parametric knowledge).
   2. Absolute factual enforcement: Every claim must cite an associated article bracket.
   3. Anti-hallucination out-of-scope breaker: Returns fallback copy if candidate quality drops.""")

if __name__ == "__main__":
    execute_telemetry_analysis("can I ask a company to delete my data and what article applies")