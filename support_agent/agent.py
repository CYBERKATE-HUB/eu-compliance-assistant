# support_agent/agent.py
import os
import json
import time
from dotenv import load_dotenv
from mistralai.client import Mistral as MistralClient
from support_agent.tools import classify_ticket, search_gdpr_kb, get_similar_resolved_tickets

load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

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
- Mark your final response as DRAFT: at the beginning"""

# Tool definitions for Mistral
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": """Classifies a GDPR support ticket into category, subcategory and urgency. Use when: a new ticket arrives — always call this FIRST before any other tool. Do NOT use when: ticket is already classified. Returns: category, subcategory, urgency, confidence, and one-sentence reasoning.""",
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
            "description": """Searches the GDPR and EU AI Act knowledge base for relevant legal articles. Use when: you need the legal basis to answer a compliance question;- When searching, use specific GDPR terminology: "right to erasure", "data subject request", "controller obligations" Do NOT use when: the question is not about GDPR or EU AI Act compliance. Returns: list of matching articles with article number, title, source, and full text.""",
            "parameters": {
                "type": "object",
                 "properties": {
                     "query": {"type": "string", "description": "Specific compliance question"},
                     "top_k": {"type": "integer", "description": "Number of articles to retrieve", "default": 3},
                     "subcategory": {"type": "string", "description": "Subcategory from classify_ticket result e.g. right_to_erasure — use this to improve search precision"}
                     },
                     "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_resolved_tickets",
            "description": """Retrieves similar previously resolved support tickets and their resolutions. Use when: classification confidence is high and you want to inform the draft with proven solutions. Do NOT use when: the ticket category is unusual or confidence is low — similar tickets may mislead. Returns: list of similar tickets with their category, subcategory, and resolution text.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_description": {"type": "string", "description": "Brief description of current ticket"},
                    "top_k": {"type": "integer", "description": "Number of similar tickets", "default": 2}
                },
                "required": ["ticket_description"]
            }
        }
    }
]

# Map tool names to actual functions
TOOL_MAP = {
    "classify_ticket": classify_ticket,
    "search_gdpr_kb": search_gdpr_kb,
    "get_similar_resolved_tickets": get_similar_resolved_tickets
}


def run_support_agent(ticket_text: str, max_iterations: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New ticket:\n{ticket_text}"}
    ]

    print(f"\n{'='*60}")
    print(f"TICKET: {ticket_text[:80]}...")
    print('='*60)

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}]")
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"API ERROR: {e}")
            return f"ESCALATE_TO_HUMAN: API error — {e}"

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # If no tool call — Mistral is done, return final answer
        if finish_reason == "stop" or not message.tool_calls:
            print("\n[Agent finished — returning draft]")
            return message.content

        # Process tool calls
        messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  → Calling: {tool_name}({list(tool_args.keys())})")

            # Execute the tool
            result = TOOL_MAP[tool_name](**tool_args)

            print(f"  ← Result: {str(result)[:100]}...")

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
            time.sleep(4)

    return "ESCALATE_TO_HUMAN: Maximum iterations reached without resolution."


if __name__ == "__main__":
    print("GDPR Support Agent ready. Type your ticket or 'quit' to exit.\n")

    while True:
        ticket = input("Ticket: ").strip()
        if ticket.lower() in ("quit", "exit", "q"):
            break
        if not ticket:
            continue
        result = run_support_agent(ticket)
        print(f"\n{result}\n")