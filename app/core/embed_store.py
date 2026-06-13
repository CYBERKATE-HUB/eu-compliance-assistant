# embed_store.py
import json
from sentence_transformers import SentenceTransformer
import chromadb

with open("all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks from all_chunks.json")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

client = chromadb.PersistentClient(path="./gdpr_db")
collection = client.get_or_create_collection(
    name="gdpr_articles",
    metadata={"hnsw:space": "cosine"}
)

print("Embedding and storing...")

for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk["text"]).tolist()
    collection.add(
        ids=[f"article_{i}"],
        embeddings=[embedding],
        documents=[chunk["text"]],
        metadatas=[{
            "article_number": chunk["article_number"],
            "article_title": chunk["article_title"],
            "source": chunk["source"],
            "language": chunk["language"]
        }]
    )
    if (i + 1) % 50 == 0:
        print(f"  Stored {i + 1}/{len(chunks)}...")

print(f"\nDone! All {len(chunks)} chunks stored.")

print("\nSanity check:")
query_embedding = model.encode("right to delete personal data").tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["metadatas", "distances"]
)
for j in range(len(results["ids"][0])):
    m = results["metadatas"][0][j]
    score = round(1 - results["distances"][0][j], 3)
    print(f"  [{score}] [{m['source']}/{m['language']}] {m['article_number']} — {m['article_title']}")