"""LLM-as-judge for evaluating draft reply quality.

Scores drafts on 5 dimensions: groundedness, brand_voice, actionability,
empathy, safety. Also detects hallucinated facts and overall sendability.

Uses Google Gemini when GEMINI_API_KEY is set, otherwise falls back to
a transparent rule-based judge.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def judge_llm(
    customer: str,
    intent: str,
    draft_reply: str,
    evidence: list[dict],
) -> dict:
    """Score a draft reply using the Gemini LLM judge."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if "gemini-2.0-flash" in model_name:
        model_name = "gemini-3.6-flash"

    template = (PROMPTS_DIR / "judge_v2.txt").read_text(encoding="utf-8")

    evidence_str = ""
    if evidence:
        for i, ex in enumerate(evidence, 1):
            evidence_str += (
                f"\nExample {i}:\n"
                f"  Customer: {ex.get('customer', '')[:150]}\n"
                f"  Reply: {ex.get('brand_reply', ex.get('reply', ''))[:150]}\n"
            )
    else:
        evidence_str = "(No evidence used)"

    prompt = (
        template
        .replace("{CUSTOMER}", customer)
        .replace("{INTENT}", intent)
        .replace("{DRAFT}", draft_reply)
        .replace("{EVIDENCE}", evidence_str)
    )

    gen_config = genai.types.GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config=gen_config)
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "not available" in str(e).lower():
            fallback_model = "gemini-3.6-flash" if model_name != "gemini-3.6-flash" else "gemini-2.5-flash"
            model = genai.GenerativeModel(fallback_model)
            response = model.generate_content(prompt, generation_config=gen_config)
        else:
            raise e

    try:
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        result = {
            "groundedness": 3, "brand_voice": 3, "actionability": 3,
            "empathy": 3, "safety": 3, "hallucinated_facts": [],
            "overall_sendable": False, "rationale": "Failed to parse judge output",
        }
    return result


def judge_rule_based(
    customer: str,
    intent: str,
    draft_reply: str,
    evidence: list[dict],
) -> dict:
    """Simple rule-based judge as a fallback when no API key is available.

    Checks for basic quality signals. This is deliberately conservative
    and labeled as rule-based in all outputs.
    """
    scores = {
        "groundedness": 4,
        "brand_voice": 3,
        "actionability": 3,
        "empathy": 3,
        "safety": 4,
    }
    hallucinated = []

    # Check if draft contains numbers not in customer message
    draft_numbers = set(re.findall(r"\b\d+(?:[.,/]\d+)*\b", draft_reply))
    customer_numbers = set(re.findall(r"\b\d+(?:[.,/]\d+)*\b", customer))
    invented = draft_numbers - customer_numbers
    if invented:
        scores["groundedness"] = 2
        hallucinated = list(invented)

    # Check for action words → actionability
    action_words = ["dm", "direct message", "try", "check", "visit", "go to", "restart"]
    if any(w in draft_reply.lower() for w in action_words):
        scores["actionability"] = 4

    # Check for empathy words
    empathy_words = ["sorry", "understand", "appreciate", "help", "important"]
    empathy_count = sum(1 for w in empathy_words if w in draft_reply.lower())
    scores["empathy"] = min(5, 2 + empathy_count)

    # Check for dangerous patterns
    danger_words = ["refund issued", "we've fixed", "resolved", "credit applied"]
    if any(w in draft_reply.lower() for w in danger_words):
        scores["safety"] = 1

    # Length check
    if len(draft_reply) > 280:
        scores["brand_voice"] = max(1, scores["brand_voice"] - 1)

    avg = sum(scores.values()) / len(scores)
    sendable = avg >= 3.5 and scores["safety"] >= 3 and scores["groundedness"] >= 3

    return {
        **scores,
        "hallucinated_facts": hallucinated,
        "overall_sendable": sendable,
        "rationale": "rule-based judge (no Gemini API configured)",
        "judge_type": "rule_based",
    }


def judge(
    customer: str,
    intent: str,
    draft_reply: str,
    evidence: list[dict],
    provider: str | None = None,
) -> dict:
    """Score a draft using the configured judge (Gemini LLM or rule-based fallback)."""
    active_provider = provider or os.getenv("PROVIDER", "heuristic")
    if active_provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        try:
            result = judge_llm(customer, intent, draft_reply, evidence)
            result["judge_type"] = "llm"
            return result
        except Exception:
            pass
    return judge_rule_based(customer, intent, draft_reply, evidence)
