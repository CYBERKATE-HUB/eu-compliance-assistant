# retrieval.py (Updated with Two-Stage Cross-Encoder Reranking)
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb

with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# --- Stage 1: Fast Retrievers (Bi-Encoder + Sparse Index) ---
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./gdpr_db")
collection = chroma_client.get_or_create_collection(
    name="gdpr_articles",
    metadata={"hnsw:space": "cosine"}
)

tokenized_chunks = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# --- Stage 2: Deep Reranker (Cross-Encoder) ---
# BAAI/bge-reranker-base is highly accurate and handles multilingual data well
reranker = CrossEncoder("BAAI/bge-reranker-base")

def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def advanced_hybrid_retrieve(query: str, over_fetch_k=15, final_top_n=3, language=None):
    """
    Two-Stage Hybrid Search Pipeline:
    1. Over-fetch candidates using fast Hybrid Search (RRF).
    2. Deeply score and reorder candidates using a Cross-Encoder Reranker.
    """
    # 1. Gather Dense Candidates from ChromaDB
    query_vector = bi_encoder.encode(query).tolist()
    filter_dict = {"language": language} if language else {}
    
    dense_res = collection.query(
        query_embeddings=[query_vector],
        n_results=over_fetch_k,
        **({"where": filter_dict} if filter_dict else {})
    )
    
    dense_ids = dense_res["ids"][0] if dense_res["ids"] else []

    # 2. Gather Sparse Candidates from BM25
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    
    filtered_indices = [
        i for i, c in enumerate(chunks) 
        if not language or c["language"] == language
    ]
    bm25_ranked = sorted(filtered_indices, key=lambda i: bm25_scores[i], reverse=True)[:over_fetch_k]
    sparse_ids = [f"article_{i}" for i in bm25_ranked]

    # 3. Fuse Rankings via RRF to establish a primary candidate pool
    fused_candidates = reciprocal_rank_fusion(dense_ids, sparse_ids)
    
    # Isolate candidate chunks for the second stage
    candidate_chunks = []
    for doc_id, _ in fused_candidates:
        idx = int(doc_id.split("_")[1])
        candidate_chunks.append(chunks[idx])

    if not candidate_chunks:
        return []

    # 4. Stage 2: Cross-Encoder Inference Reranking
    # Pair the single user query with every unique candidate text string
    rerank_pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(rerank_pairs)

    # Attach structural scores back to the source objects
    for i, score in enumerate(rerank_scores):
        candidate_chunks[i]["rerank_score"] = float(score)

    # Sort final output based entirely on real-time textual relevance alignment
    final_sorted_chunks = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return final_sorted_chunks[:final_top_n]

def get_article_by_number(number, source=None, language=None):
    """Direct lookup by article number — bypasses semantic search."""
    matches = []
    for chunk in chunks:
        if chunk["article_number"].lower() == number.lower():
            if source and chunk["source"] != source:
                continue
            if language and chunk["language"] != language:
                continue
            matches.append(chunk)
    return matches