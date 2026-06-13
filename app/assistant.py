# assistant.py
import os
import re
from dotenv import load_dotenv
from mistralai.client import Mistral as MistralClient
from app.core.retrieval import advanced_hybrid_retrieve as hybrid_retrieve, get_article_by_number
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

client = MistralClient(api_key=api_key)

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path="./gdpr_db")
collection = chroma.get_or_create_collection(
    name="gdpr_articles",
    metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = """You are an EU law compliance assistant covering GDPR and the AI Act.

Rules:
- Answer ONLY using the provided context articles. No outside knowledge.
- Every factual claim must cite a specific Article number in brackets, e.g. [Article 17].
- If the context does not contain enough information, say explicitly:
  "The provided articles do not cover this in sufficient detail."
- Do not speculate. Do not invent articles.

Format: Clear prose with article citations in brackets at the end of each claim."""


def answer(query: str) -> str:
    french_indicators = ["quel", "quels", "quelle", "quelles", "est", "sont", "les", "des", "une", "pour", "dans"]
    language = "fr" if any(w in query.lower().split() for w in french_indicators) else "en"

    article_match = re.search(r'article\s+(\d+)', query.lower())
    if article_match:
        article_num = f"Article {article_match.group(1)}"
        direct_matches = get_article_by_number(article_num, language=language)
        if direct_matches:
            chunks = direct_matches + hybrid_retrieve(query, final_top_n=2, language=language)
        else:
            chunks = hybrid_retrieve(query, final_top_n=3, language=language)
    else:
        chunks = hybrid_retrieve(query, final_top_n=3, language=language)

    if not chunks:
        return "Could not retrieve relevant articles for this question."

    query_embedding = model.encode(query).tolist()
    check = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["distances"]
    )
    top_score = 1 - check["distances"][0][0]
    if top_score < 0.4:
        return "I don't have sufficient information to answer this question reliably."

    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[{chunk['article_number']} / {chunk['source']} / {chunk['language']}] "
            f"{chunk['article_title']}\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context articles:\n\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("EU Compliance Assistant ready (GDPR + AI Act, EN + FR). Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not query:
            continue
        print("\nAssistant: ", end="")
        print(answer(query))
        print()