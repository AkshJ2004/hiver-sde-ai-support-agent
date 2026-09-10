import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
LONG_DIGITS = re.compile(r"(?<!\d)\d{7,}(?!\d)")

def scrub_pii(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = PHONE.sub("[REDACTED_PHONE]", text)
    return LONG_DIGITS.sub("[REDACTED_ID]", text)

def has_order_reference(text: str) -> bool:
    return bool(re.search(r"\b(order|tracking|shipment|package)\b|\b\d{4,}\b", text, re.I))

def high_frustration(text: str) -> bool:
    return bool(re.search(r"\b(furious|angry|worst|unacceptable|scam|fraud|lawsuit|hate)\b|!{2,}", text, re.I))

def requests_human(text: str) -> bool:
    return bool(re.search(r"\b(human\s+agent|real\s+person|talk\s+to\s+(?:a\s+)?human|speak\s+to\s+(?:a\s+)?human|live\s+agent|representative|manager)\b", text, re.I))

UNAUTHORIZED_COMMITMENTS = re.compile(
    r"\b(refund\s+(?:issued|approved|processed)|credit\s+(?:applied|issued)|"
    r"free\s+replacement|free\s+repair|warranty\s+(?:extended|approved)|"
    r"credited\s+your\s+account|\$\d+)\b",
    re.IGNORECASE,
)

def unverifiable_claim(draft: str, customer: str = "") -> bool:
    """Detect unauthorized financial, warranty, or legal commitments in AI drafts."""
    return bool(UNAUTHORIZED_COMMITMENTS.search(draft))

