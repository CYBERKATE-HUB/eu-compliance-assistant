# support_agent/tools.py
import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral as MistralClient
from app.core.retrieval import advanced_hybrid_retrieve
import chromadb

load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))


def classify_ticket(ticket_text: str) -> dict:
    """
    Classifies a GDPR support ticket.
    Returns category, subcategory, urgency, and confidence.
    """
    prompt = f"""Classify this GDPR support ticket. Return ONLY a JSON object, no other text.

Ticket:
{ticket_text}

Return this exact JSON structure:
{{
    "category": "one of: data_subject_request | data_breach | consent | third_party_sharing | retention | other",
    "subcategory": "specific issue e.g. right_to_erasure | right_to_access | breach_notification | withdrawal_of_consent",
    "urgency": "high | medium | low",
    "confidence": "high | medium | low",
    "reasoning": "one sentence explaining the classification"
}}"""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())

GDPR_QUERY_MAP = {
    "right_to_erasure": "right to erasure data subject request Article 17",
    "right_to_access": "right of access data subject Article 15",
    "right_to_portability": "right to data portability Article 20",
    "breach_notification": "personal data breach notification supervisory authority Article 33",
    "withdrawal_of_consent": "withdrawal of consent processing Article 7",
    "international_transfer": "transfer personal data third country Article 44 45 46",
}

def search_gdpr_kb(query: str, top_k: int = 3, subcategory: str = None) -> list:
    # Use known GDPR terminology if subcategory is provided
    enriched = GDPR_QUERY_MAP.get(subcategory, query) if subcategory else query
    return advanced_hybrid_retrieve(enriched, final_top_n=top_k, language=None)
   
def get_similar_resolved_tickets(ticket_description: str, top_k: int = 2) -> list:
    """
    Retrieves similar previously resolved tickets from memory.
    """
    chroma = chromadb.PersistentClient(path="./data/gdpr_db")
    collection = chroma.get_or_create_collection(
        name="resolved_tickets",
        metadata={"hnsw:space": "cosine"}
    )

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = model.encode(ticket_description).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    similar = []
    for i in range(len(results["ids"][0])):
        similar.append({
            "ticket": results["documents"][0][i],
            "category": results["metadatas"][0][i]["category"],
            "subcategory": results["metadatas"][0][i]["subcategory"],
            "resolution": results["metadatas"][0][i]["resolution"],
            "similarity": round(1 - results["distances"][0][i], 3)
        })

    return similar