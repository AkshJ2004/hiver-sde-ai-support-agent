"""Core agent pipeline: classify → retrieve → draft → decide.

Supports three modes:
  - heuristic: keyword classify + template draft (simple baseline)
  - majority:  always-majority classify + template draft (trivial baseline)
  - gemini:    LLM classify + RAG draft via Google Gemini (full system)

Set PROVIDER env var to switch. Default is heuristic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .constants import ACTIONABLE, CONF_THRESHOLD, INTENTS, SENSITIVE
from .provider import classify, classify_heuristic, classify_majority
from .safety import has_order_reference, high_frustration, requests_human, unverifiable_claim

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# ── Template drafts (for heuristic / majority baselines) ─────────────
TEMPLATE_DRAFTS: dict[str, str] = {
    "software_update_issue": (
        "Hi, thanks for reaching out. We'd like to help with the update issue. "
        "Please DM us your device model and iOS version so we can look into this together."
    ),
    "battery_or_power": (
        "Hi, battery life is important and we're here for you. "
        "Please DM us your device model and iOS version. We'll go from there."
    ),
    "app_or_performance": (
        "Hi, we'd love to help. Please DM us which apps are affected "
        "and the steps you've already tried. We'll take a look."
    ),
    "connectivity": (
        "Hi, we'd like to help with the connection issue. "
        "Please DM us your device info and what you've tried so far."
    ),
    "account_or_security": (
        "Hi, we understand this is important. For your security, "
        "please DM us so we can help you regain access to your account."
    ),
    "general_inquiry": (
        "Hi, thanks for reaching out! We'd be glad to help. "
        "For accessories, replacement parts, or general questions, you can check apple.com or your nearest Apple Store. "
        "Feel free to send us a DM if you'd like more personalized assistance with your device!"
    ),
}


# ── Retrieval ────────────────────────────────────────────────────────
def retrieve(
    text: str,
    intent: str,
    kb: list[dict],
    brand: str | None = None,
    k: int = 3,
) -> list[dict]:
    """Retrieve similar historical examples by intent + token overlap.

    Filters to same intent and brand, excludes the input itself,
    and scores by normalized token overlap.
    """
    tokens = set(text.lower().split())
    options = [
        x for x in kb
        if x.get("intent") == intent
        and (brand is None or x.get("brand") == brand)
        and x["customer"].strip() != text.strip()
    ]
    scored = []
    for x in options:
        x_tokens = set(x["customer"].lower().split())
        overlap = len(tokens & x_tokens)
        score = overlap / max(len(tokens), 1)
        scored.append((score, x))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [x for _, x in scored[:k]]


# ── Draft generation ─────────────────────────────────────────────────
def draft_template(text: str, intent: str, evidence: list[dict]) -> str:
    """Template-based drafting — the baseline approach."""
    return TEMPLATE_DRAFTS.get(intent, TEMPLATE_DRAFTS["general_inquiry"])


def draft_llm(text: str, intent: str, evidence: list[dict]) -> str:
    """LLM-powered RAG draft grounded in retrieved evidence (Gemini)."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if "gemini-2.0-flash" in model_name:
        model_name = "gemini-3.6-flash"

    template = (PROMPTS_DIR / "draft_v2.txt").read_text(encoding="utf-8")

    # Format evidence for the prompt
    evidence_str = ""
    if evidence:
        for i, ex in enumerate(evidence, 1):
            evidence_str += (
                f"\nExample {i}:\n"
                f"  Customer: {ex['customer']}\n"
                f"  AppleSupport reply: {ex['brand_reply']}\n"
            )
    else:
        evidence_str = "(No similar historical examples found)"

    prompt = (
        template
        .replace("{CUSTOMER}", text)
        .replace("{INTENT}", intent)
        .replace("{EVIDENCE}", evidence_str)
    )

    gen_config = genai.types.GenerationConfig(
        temperature=0.3,
        max_output_tokens=150,
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config=gen_config)
        return response.text.strip()
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "not available" in str(e).lower():
            try:
                fallback_model = "gemini-3.6-flash" if model_name != "gemini-3.6-flash" else "gemini-2.5-flash"
                model = genai.GenerativeModel(fallback_model)
                response = model.generate_content(prompt, generation_config=gen_config)
                return response.text.strip()
            except Exception:
                return draft_template(text, intent, evidence)
        else:
            return draft_template(text, intent, evidence)


def draft(text: str, intent: str, evidence: list[dict]) -> str:
    """Draft a reply using the configured provider."""
    provider = os.getenv("PROVIDER", "heuristic")
    if provider == "gemini":
        return draft_llm(text, intent, evidence)
    return draft_template(text, intent, evidence)


# ── Adaptive situational threshold calculation ──────────────────────
def calculate_adaptive_threshold(intent: str, customer: str) -> tuple[float, str, str]:
    """Dynamically determine the confidence threshold based on situational risk.

    Returns:
        (threshold_value, risk_level, explanation)

    Situational risk tiers:
    - High Risk (0.85): Security, credentials, Apple ID lock, financial/billing.
      Strict threshold required to prevent incorrect guidance on sensitive accounts.
    - Elevated Risk (0.80): Customer displaying frustration, anger, or strong urgency.
      Conservative threshold to ensure upset customers receive vetted handling.
    - Medium Risk (0.68): Standard technical troubleshooting (battery, connectivity,
      software updates, app performance).
    - Low Risk (0.55): Routine queries, cases, covers, chargers, store hours,
      accessory compatibility. Safe for automated assistance without false escalations.
    """
    customer_lower = customer.lower()

    # 1. High Risk: Account, Security, Identity, Billing
    if intent in SENSITIVE or any(
        k in customer_lower
        for k in [
            "hacked", "stolen", "password", "apple id", "passcode", "compromised",
            "unauthorized", "charge", "refund", "billing", "credit card", "bank"
        ]
    ):
        return (
            0.85,
            "High Risk (Security / Identity / Billing)",
            "Strict safety margin applied: sensitive account, credential, or financial inquiry.",
        )

    # 2. Elevated Risk: Frustration / Urgent Escalation Tone
    if high_frustration(customer) or any(
        k in customer_lower
        for k in [
            "ridiculous", "unacceptable", "terrible", "awful", "horrible",
            "worst customer", "fix this now", "lawyer", "furious", "disgusted"
        ]
    ):
        return (
            0.80,
            "Elevated Risk (Customer Frustration)",
            "Cautionary safety margin applied: customer exhibits elevated frustration or urgency.",
        )

    # 3. Low Risk: Routine Inquiries, Accessories, Store, Cases, Cables
    if intent == "general_inquiry" or any(
        k in customer_lower
        for k in [
            "case", "cover", "charger", "cable", "adapter", "store", "buy", "shop",
            "price", "scratch", "accessory", "accessories", "screen protector",
            "trade-in", "retail", "hours", "available", "color", "compatible"
        ]
    ):
        return (
            0.55,
            "Low Risk (Routine Inquiry & Accessories)",
            "Adaptive baseline threshold applied: safe, routine hardware or retail question.",
        )

    # 4. Medium Risk: Standard Technical Troubleshooting
    return (
        0.68,
        "Medium Risk (Technical Troubleshooting)",
        "Standard technical support threshold applied for device troubleshooting.",
    )


# ── Escalation decision ─────────────────────────────────────────────
def decide(
    intent: str,
    confidence: float,
    customer: str,
    draft_text: str,
    threshold: float | str | None = "auto",
) -> tuple[str, str]:
    """Deterministic escalation policy.

    Returns (decision, reason) where decision is 'auto_handle' or 'escalate'.
    The policy is conservative: when in doubt, escalate.
    If threshold is 'auto' or None, dynamically calibrates situational threshold.
    """
    if threshold == "auto" or threshold is None:
        effective_threshold, risk_level, _ = calculate_adaptive_threshold(intent, customer)
    else:
        try:
            effective_threshold = float(threshold)
            risk_level = "Manual Override"
        except (ValueError, TypeError):
            effective_threshold, risk_level, _ = calculate_adaptive_threshold(intent, customer)

    if intent in SENSITIVE:
        return "escalate", "policy: identity/security intents always require human review"
    if requests_human(customer):
        return "escalate", "customer explicitly requested a human agent"
    if high_frustration(customer):
        return "escalate", "elevated customer frustration or escalation tone detected"
    if confidence < effective_threshold:
        return "escalate", f"low classifier confidence ({confidence:.2f} < {effective_threshold:.2f} [{risk_level}])"
    if unverifiable_claim(draft_text, customer):
        return "escalate", "draft contains an unauthorized financial/hardware commitment"
    return "auto_handle", f"high-confidence, non-sensitive, verifiable resolution draft ({confidence:.2f} >= {effective_threshold:.2f})"


# ── Main inference entry point ───────────────────────────────────────
def infer(
    customer: str,
    kb: list[dict],
    brand: str | None = None,
    threshold: float | str | None = "auto",
    provider_override: str | None = None,
) -> dict:
    """Run the full pipeline on a single customer message.

    Returns a dict with intent, confidence, draft_reply, decision,
    decision_reason, effective_threshold, risk_tier, and retrieved_evidence.
    """
    original_provider = os.environ.get("PROVIDER", "heuristic")
    if provider_override:
        os.environ["PROVIDER"] = provider_override

    try:
        result = classify(customer)
        intent = result["intent"]
        confidence = result["confidence"]

        if threshold == "auto" or threshold is None:
            eff_threshold, risk_tier, thresh_desc = calculate_adaptive_threshold(intent, customer)
            thresh_mode = "auto"
        else:
            try:
                eff_threshold = float(threshold)
                risk_tier = "Manual Override"
                thresh_desc = f"Fixed manual confidence threshold ({eff_threshold:.2f})"
                thresh_mode = "manual"
            except (ValueError, TypeError):
                eff_threshold, risk_tier, thresh_desc = calculate_adaptive_threshold(intent, customer)
                thresh_mode = "auto"

        evidence = retrieve(customer, intent, kb, brand)
        reply = draft(customer, intent, evidence)
        decision, reason = decide(intent, confidence, customer, reply, eff_threshold)

        return {
            "intent": intent,
            "confidence": confidence,
            "brand": brand,
            "draft_reply": reply,
            "decision": decision,
            "decision_reason": reason,
            "effective_threshold": eff_threshold,
            "risk_tier": risk_tier,
            "threshold_explanation": thresh_desc,
            "threshold_mode": thresh_mode,
            "retrieved_evidence": [
                {"id": x["id"], "customer": x["customer"][:100], "reply": x["brand_reply"][:100]}
                for x in evidence
            ],
            "provider": os.environ.get("PROVIDER", "heuristic"),
        }
    finally:
        os.environ["PROVIDER"] = original_provider

