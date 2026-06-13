# parse_documents.py
from bs4 import BeautifulSoup
import json

DOCUMENTS = [
    {"file": "gdpr.html",       "source": "gdpr",   "language": "en"},
    {"file": "gdpr_fr.html",    "source": "gdpr",   "language": "fr"},
    {"file": "ai_act_en.html",  "source": "ai_act", "language": "en"},
    {"file": "ai_act_fr.html",  "source": "ai_act", "language": "fr"},
]

def parse_document(file, source, language):
    with open(file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    chunks = []
    all_tags = soup.find_all(["h2", "h3", "h4", "h5", "p"])

    current_article = None
    current_text = []
    current_title = ""
    current_number = ""

    for tag in all_tags:
        text = tag.get_text(strip=True).replace("\xa0", " ").strip()

        if text.startswith("Article ") and len(text) < 60:
            if current_article and current_text:
                chunks.append({
                    "article_number": current_number,
                    "article_title": current_title,
                    "text": " ".join(current_text),
                    "source": source,
                    "language": language
                })
            current_number = text.strip()
            current_title = ""
            current_text = []
            current_article = "just_started"

        elif current_article == "just_started" and text:
            current_title = text
            current_article = True

        elif current_article:
            if text:
                current_text.append(text)

    if current_article and current_text:
        chunks.append({
            "article_number": current_number,
            "article_title": current_title,
            "text": " ".join(current_text),
            "source": source,
            "language": language
        })

    return chunks


all_chunks = []

for doc in DOCUMENTS:
    print(f"Parsing {doc['file']}...")
    chunks = parse_document(doc["file"], doc["source"], doc["language"])
    all_chunks.extend(chunks)
    print(f"  Found {len(chunks)} articles ({doc['source']} / {doc['language']})")

print(f"\nTotal chunks: {len(all_chunks)}")

with open("all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("Saved to all_chunks.json")

print("\nSample from each document:")
seen = set()
for c in all_chunks:
    key = f"{c['source']}_{c['language']}"
    if key not in seen:
        seen.add(key)
        print(f"\n  [{c['source']} / {c['language']}]")
        print(f"  {c['article_number']} — {c['article_title']}")
        print(f"  {c['text'][:80]}...")