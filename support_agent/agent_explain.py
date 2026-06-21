# support_agent/agent_explain.py
# Shows every internal step of the support agent for a given ticket.

import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral as MistralClient
from sentence_transformers import SentenceTransformer
from support_agent.tools import classify_ticket, search_gdpr_kb, get_similar_resolved_tickets
from app.core.retrieval import advanced_hybrid_retrieve
import chromadb

load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": """Classifies a GDPR support ticket into category, subcategory and urgency.
Use when: a new ticket arrives — always call this FIRST before any other tool.
Do NOT use when: ticket is already classified.
Returns: category, subcategory, urgency, confidence, and one-sentence reasoning.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {"type": "string", "description": "The full ticket content"}
                },
                "required": ["ticket_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_gdpr_kb",
            "description": """Searches the GDPR and EU AI Act knowledge base for relevant legal articles.
Use when: you need the legal basis to answer a compliance question.
Do NOT use when: the question is not about GDPR or EU AI Act compliance.
Returns: list of matching articles with article number, title, source, and full text.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "subcategory": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_resolved_tickets",
            "description": """Retrieves similar previously resolved support tickets and their resolutions.
Use when: classification confidence is high and you want to inform the draft with proven solutions.
Do NOT use when: the ticket category is unusual or confidence is low.
Returns: list of similar tickets with category, subcategory, and resolution text.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_description": {"type": "string"},
                    "top_k": {"type": "integer", "default": 2}
                },
                "required": ["ticket_description"]
            }
        }
    }
]

TOOL_MAP = {
    "classify_ticket": classify_ticket,
    "search_gdpr_kb": search_gdpr_kb,
    "get_similar_resolved_tickets": get_similar_resolved_tickets
}

SYSTEM_PROMPT = """You are a GDPR compliance support agent.

For each ticket you must:
1. First classify the ticket using classify_ticket
2. Search for relevant GDPR articles using search_gdpr_kb — call this MAXIMUM ONCE OR TWICE
3. Optionally retrieve similar past resolutions using get_similar_resolved_tickets — call MAXIMUM ONCE
4. Draft a response that cites specific GDPR articles

Rules:
- Always classify before searching
- Call search_gdpr_kb maximum 2 times total — do not repeat searches
- After classifying and searching once, you have enough context to write a draft
- Every response must cite at least one GDPR article
- If you cannot find relevant articles after searching, write the best draft you can and mark uncertain parts
- If the question is completely outside GDPR/AI Act scope, say ESCALATE_TO_HUMAN and explain why
- Never invent legal requirements — only cite retrieved articles
- Mark your final response as DRAFT: at the beginning
- When searching, use specific GDPR terminology: right to erasure, data subject request, controller obligations"""


def sep(title, char="="):
    print(f"\n{char*60}")
    print(f"  {title}")
    print(char*60)


def explain_agent(ticket_text: str):

    # ─────────────────────────────────────────────
    sep("STEP 1 — TICKET RECEIVED")
    # ─────────────────────────────────────────────
    print(f"""
  Ticket: "{ticket_text[:100]}..."
  Length: {len(ticket_text)} characters
  Words : {len(ticket_text.split())} words
""")

    # ─────────────────────────────────────────────
    sep("STEP 2 — TICKET EMBEDDING")
    # ─────────────────────────────────────────────
    ticket_vector = model.encode(ticket_text)
    print(f"""
  Model    : all-MiniLM-L6-v2
  Why      : even though we classify with Mistral, the ticket is
             also embedded for similarity search in memory store.

  Vector preview (first 8 of 384 dimensions):
  {ticket_vector[:8].tolist()}

  This vector is used by get_similar_resolved_tickets to find
  past tickets with similar meaning — not exact word match.
""")

    # ─────────────────────────────────────────────
    sep("STEP 3 — AGENT LOOP (ReAct pattern)")
    # ─────────────────────────────────────────────
    print("""
  Pattern : Reason → Act → Observe → Repeat until done
  Max iterations: 5
  Tools available: classify_ticket, search_gdpr_kb, get_similar_resolved_tickets

  Each iteration:
    1. Mistral sees full message history + tool results so far
    2. Decides: call a tool OR write final answer
    3. If tool: we execute it, add result to history
    4. If no tool call (finish_reason=stop): done
""")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New ticket:\n{ticket_text}"}
    ]

    iteration_log = []

    for iteration in range(5):
        sep(f"ITERATION {iteration + 1}", char="-")

        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"  API ERROR: {e}")
            break

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"\n  finish_reason: {finish_reason}")

        if finish_reason == "stop" or not message.tool_calls:
            print("""
  → Mistral decided: I have enough context to write the final answer.
  → No more tool calls needed.
  → Writing DRAFT response now.
""")
            iteration_log.append({
                "iteration": iteration + 1,
                "action": "FINAL ANSWER",
                "reasoning": "finish_reason=stop — agent has sufficient context"
            })
            break

        # Show reasoning
        print(f"\n  Mistral's decision this iteration:")
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"  → Call: {tc.function.name}")
            print(f"     Args: {args}")

            # Why did it choose this tool?
            reasoning = {
                "classify_ticket": "First step always — need to understand what type of issue this is before searching.",
                "search_gdpr_kb": "Need legal basis — searching knowledge base for relevant GDPR/AI Act articles.",
                "get_similar_resolved_tickets": "Classification done — checking memory for similar resolved cases."
            }
            print(f"     Why : {reasoning.get(tc.function.name, 'agent decision')}")

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"\n  Executing {tool_name}...")
            result = TOOL_MAP[tool_name](**tool_args)

            # Show results differently per tool
            if tool_name == "classify_ticket":
                print(f"""
  Classification result:
    category   : {result.get('category')}
    subcategory: {result.get('subcategory')}
    urgency    : {result.get('urgency')}
    confidence : {result.get('confidence')}
    reasoning  : {result.get('reasoning')}

  Why this matters: the subcategory guides the KB search query.
  A wrong classification here means wrong articles retrieved.
""")

            elif tool_name == "search_gdpr_kb":
                print(f"\n  KB search results ({len(result)} articles):")
                for r in result:
                    print(f"    [{r['article_number']}] {r['article_title']} ({r['source']}/{r['language']})")
                    print(f"    Preview: {r['text'][:80]}...")
                print(f"""
  Retrieval method: hybrid (dense cosine + BM25) merged with RRF
  These articles become the grounding context for the draft.
""")

            elif tool_name == "get_similar_resolved_tickets":
                print(f"\n  Similar tickets found ({len(result)}):")
                for r in result:
                    print(f"    [{r['similarity']}] {r['subcategory']}")
                    print(f"    Ticket    : {r['ticket'][:80]}...")
                    print(f"    Resolution: {r['resolution'][:80]}...")
                print(f"""
  These past resolutions inform the draft structure.
  High similarity (>0.8) = strong signal to follow same resolution pattern.
""")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

            iteration_log.append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "args": tool_args
            })

        time.sleep(4)

    # ─────────────────────────────────────────────
    sep("STEP 4 — FINAL DRAFT")
    # ─────────────────────────────────────────────
    final_response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages,
        tools=TOOLS,
        tool_choice="none"
    )
    draft = final_response.choices[0].message.content

    print(f"\n{draft}")

    # ─────────────────────────────────────────────
    sep("STEP 5 — AGENT TRACE SUMMARY")
    # ─────────────────────────────────────────────
    print(f"\n  Total iterations: {len(iteration_log)}")
    for log in iteration_log:
        if "tool" in log:
            print(f"  [{log['iteration']}] {log['tool']}({list(log['args'].keys())})")
        else:
            print(f"  [{log['iteration']}] {log['action']}")

    print(f"""
  Key architectural decisions illustrated:
  1. Classify first  → narrows search space, improves retrieval precision
  2. ReAct loop      → agent decides tool sequence, not hardcoded pipeline
  3. Memory search   → past resolutions improve draft quality
  4. Grounding prompt → draft cites only retrieved articles, no hallucination
  5. Max iterations  → safety limit prevents infinite tool loops
""")


if __name__ == "__main__":
    print("Support Agent Explainer — shows every internal step\n")
    ticket = input("Enter ticket: ").strip()
    explain_agent(ticket)