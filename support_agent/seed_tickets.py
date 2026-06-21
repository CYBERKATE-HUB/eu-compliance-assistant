# support_agent/seed_tickets.py
# Sample resolved tickets for demo purposes.

RESOLVED_TICKETS = [
    {
        "id": "ticket_001",
        "ticket": "A customer is requesting we delete all their personal data including purchase history.",
        "category": "data_subject_request",
        "subcategory": "right_to_erasure",
        "resolution": "Confirmed obligation under Article 17 GDPR. Data deleted within 30 days. Customer notified in writing per Article 12(3)."
    },
    {
        "id": "ticket_002",
        "ticket": "We discovered unauthorized access to our customer database. About 500 records may have been exposed.",
        "category": "data_breach",
        "subcategory": "breach_notification",
        "resolution": "Reported to supervisory authority within 72 hours per Article 33. Affected customers notified per Article 34. Incident logged."
    },
    {
        "id": "ticket_003",
        "ticket": "Customer wants to know what personal data we hold about them.",
        "category": "data_subject_request",
        "subcategory": "right_to_access",
        "resolution": "Subject access request fulfilled under Article 15. Full data export provided within one month per Article 12(3)."
    },
    {
        "id": "ticket_004",
        "ticket": "Customer withdrew their marketing consent and wants us to stop all communications.",
        "category": "consent",
        "subcategory": "withdrawal_of_consent",
        "resolution": "Consent withdrawal processed per Article 7(3). Removed from all marketing lists immediately. Confirmed processing was consent-based only."
    },
    {
        "id": "ticket_005",
        "ticket": "We want to share customer data with a third party analytics provider in the US.",
        "category": "third_party_sharing",
        "subcategory": "international_transfer",
        "resolution": "Transfer assessed under Article 44-46. Standard Contractual Clauses implemented. Transfer impact assessment completed."
    }
]