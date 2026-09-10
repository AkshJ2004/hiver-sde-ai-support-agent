"""Intent classification providers: heuristic (keyword) and LLM-backed (Gemini).

The heuristic provider is the *simple baseline* for evaluation — it uses
keyword matching and returns a deterministic result with no API calls.

The LLM provider uses Gemini (Google) with few-shot prompting and structured
JSON output for production-quality classification.

Set PROVIDER=gemini and GEMINI_API_KEY to activate LLM classification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .constants import INTENTS

# ── Keyword lists for the heuristic baseline ─────────────────────────
KEYWORDS: dict[str, tuple[str, ...]] = {
    "software_update_issue": (
        "update", "updated", "upgrade", "ios 11", "ios11", "ios 12", "ios12",
        "ios 10", "new software", "after the update", "since the update",
        "latest version", "new version", "firmware",
    ),
    "battery_or_power": (
        "battery", "batteries", "charge", "charging", "charger", "power",
        "drain", "dies", "dead", "overheating", "hot", "won't turn on",
        "shuts down", "shutdown",
    ),
    "app_or_performance": (
        "crash", "crashes", "crashing", "freeze", "freezes", "freezing",
        "frozen", "stuck", "slow", "lag", "laggy", "hang", "hanging",
        "glitch", "bug", "unresponsive", "loading",
    ),
    "connectivity": (
        "wifi", "wi-fi", "bluetooth", "cellular", "signal", "connection",
        "connect", "disconnect", "network", "internet", "lte", "4g", "5g",
        "airpods", "pair", "pairing",
    ),
    "account_or_security": (
        "apple id", "appleid", "password", "login", "log in", "sign in",
        "locked out", "locked", "account", "two factor", "2fa", "verification",
        "icloud", "hacked", "security",
    ),
    "general_inquiry": (
        "case", "cover", "screen", "scratch", "broken", "crack", "damage", "damaged",
        "accessory", "accessories", "cable", "cord", "charger", "charging cable", "lightning", "usb",
        "headphone", "headphones", "adapter", "pencil", "keyboard", "watch band", "strap",
        "buy", "purchase", "cost", "price", "order", "store", "shop", "shipping", "deliver", "delivery",
        "apple store", "retail", "trade in", "trade-in", "warranty", "applecare",
        "how to", "how do i", "can i", "is it possible", "question", "help", "assist", "inquiry",
        "setup", "set up", "transfer", "backup", "restore", "clean", "cleaning",
        "replace", "replacement", "new", "recommend", "compatibility",
    ),
}

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def classify_heuristic(text: str) -> dict:
    """Rule-based keyword classifier — the 'simple baseline'."""
    low = text.lower()
    scored = {i: sum(k in low for k in keys) for i, keys in KEYWORDS.items()}
    intent, score = max(scored.items(), key=lambda x: x[1])
    if not score:
        is_question = "?" in text or any(low.startswith(w) for w in ("how", "what", "where", "when", "can", "is", "could", "why"))
        if is_question or len(text.split()) >= 4:
            intent, confidence = "general_inquiry", 0.80
        else:
            intent, confidence = "general_inquiry", 0.55
    else:
        confidence = min(0.95, 0.65 + 0.12 * score)
    return {"intent": intent, "confidence": round(confidence, 3)}


def classify_majority(text: str) -> dict:
    """Always predicts the majority class — the 'trivial baseline'."""
    return {"intent": "software_update_issue", "confidence": 1.0}


def classify_llm(text: str) -> dict:
    """LLM-powered classification using Google Gemini with structured output."""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Export it or add to .env")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if "gemini-2.0-flash" in model_name:
        model_name = "gemini-3.6-flash"

    prompt_template = (PROMPTS_DIR / "classify_v2.txt").read_text(encoding="utf-8")
    allowed = ", ".join(INTENTS)
    system_prompt = prompt_template.replace("{INTENTS}", allowed)
    content_prompt = f"{system_prompt}\n\nCustomer message:\n{text}"

    gen_config = genai.types.GenerationConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(content_prompt, generation_config=gen_config)
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "not available" in str(e).lower():
            try:
                fallback_model = "gemini-3.6-flash" if model_name != "gemini-3.6-flash" else "gemini-2.5-flash"
                model = genai.GenerativeModel(fallback_model)
                response = model.generate_content(content_prompt, generation_config=gen_config)
            except Exception:
                return classify_heuristic(text)
        else:
            # On 429 quota exhaustion or network error, smoothly fall back to heuristic
            return classify_heuristic(text)

    try:
        raw = response.text.strip()
        # Handle markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"intent": "general_inquiry", "confidence": 0.5}

    # Validate intent is in taxonomy
    if result.get("intent") not in INTENTS:
        result["intent"] = "general_inquiry"
        result["confidence"] = 0.5

    return {
        "intent": result["intent"],
        "confidence": round(float(result.get("confidence", 0.7)), 3),
    }


# ── Provider dispatcher ──────────────────────────────────────────────
_PROVIDER = os.getenv("PROVIDER", "heuristic")


def classify(text: str) -> dict:
    """Classify using the configured provider."""
    provider = os.getenv("PROVIDER", "heuristic")
    if provider == "gemini":
        return classify_llm(text)
    elif provider == "majority":
        return classify_majority(text)
    else:
        return classify_heuristic(text)
