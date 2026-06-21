# support_agent/seed_memory.py
# Run once to index sample resolved tickets into ChromaDB.

from sentence_transformers import SentenceTransformer
import chromadb
from support_agent.seed_tickets import RESOLVED_TICKETS

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./data/gdpr_db")

collection = client.get_or_create_collection(
    name="resolved_tickets",
    metadata={"hnsw:space": "cosine"}
)

print("Indexing resolved tickets...")

for ticket in RESOLVED_TICKETS:
    # We embed the ticket text — this is what we'll search against
    embedding = model.encode(ticket["ticket"]).tolist()
    collection.add(
        ids=[ticket["id"]],
        embeddings=[embedding],
        documents=[ticket["ticket"]],
        metadatas=[{
            "category": ticket["category"],
            "subcategory": ticket["subcategory"],
            "resolution": ticket["resolution"]
        }]
    )
    print(f"  Indexed {ticket['id']} — {ticket['subcategory']}")

print(f"\nDone! {len(RESOLVED_TICKETS)} tickets indexed.")