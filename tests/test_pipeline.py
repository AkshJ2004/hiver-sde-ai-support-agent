"""Smoke tests for the AppleSupport copilot pipeline."""
import json
from pathlib import Path

import pytest

from pipeline.constants import INTENTS, SENSITIVE
from pipeline.provider import classify_heuristic, classify_majority
from pipeline.safety import scrub_pii, high_frustration, has_order_reference
from pipeline.core import decide, draft_template, retrieve


# ── Classification tests ─────────────────────────────────────────────

def test_classify_heuristic_returns_valid_intent():
    result = classify_heuristic("My battery drains so fast after the update")
    assert result["intent"] in INTENTS
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_majority_always_same():
    r1 = classify_majority("battery issue")
    r2 = classify_majority("wifi problem")
    assert r1["intent"] == r2["intent"]


def test_classify_battery_keywords():
    result = classify_heuristic("My iPhone battery dies in 2 hours")
    assert result["intent"] == "battery_or_power"


def test_classify_update_keywords():
    result = classify_heuristic("Since the iOS update everything is broken")
    assert result["intent"] == "software_update_issue"


def test_classify_connectivity_keywords():
    result = classify_heuristic("My wifi keeps disconnecting on my iPhone")
    assert result["intent"] == "connectivity"


def test_classify_account_keywords():
    result = classify_heuristic("I'm locked out of my Apple ID")
    assert result["intent"] == "account_or_security"


def test_classify_empty_defaults_to_general():
    result = classify_heuristic("thanks")
    assert result["intent"] == "general_inquiry"


# ── PII scrubbing tests ─────────────────────────────────────────────

def test_scrub_email():
    assert "[REDACTED_EMAIL]" in scrub_pii("contact me at user@example.com")


def test_scrub_phone():
    assert "[REDACTED_PHONE]" in scrub_pii("call me at +1 555 123 4567")


def test_scrub_long_digits():
    result = scrub_pii("order number 1234567890")
    assert "[REDACTED_" in result  # May be PHONE or ID depending on regex order


def test_scrub_preserves_normal_text():
    text = "My phone is broken please help"
    assert scrub_pii(text) == text


# ── Safety tests ─────────────────────────────────────────────────────

def test_high_frustration_detects_anger():
    assert high_frustration("This is the worst service ever!!")
    assert high_frustration("I am furious about this")


def test_high_frustration_normal_message():
    assert not high_frustration("My wifi isn't working, can you help?")


def test_has_order_reference():
    assert has_order_reference("my order 12345 hasn't arrived")
    assert not has_order_reference("my phone is slow")


# ── Escalation decision tests ───────────────────────────────────────

def test_sensitive_intents_always_escalate():
    for intent in SENSITIVE:
        decision, reason = decide(intent, 0.95, "test message", "test draft")
        assert decision == "escalate"
        assert "identity" in reason.lower() or "security" in reason.lower()


def test_low_confidence_escalates():
    decision, reason = decide("general_inquiry", 0.50, "test", "test")
    assert decision == "escalate"
    assert "confidence" in reason.lower()


def test_high_confidence_general_auto_handles():
    decision, reason = decide("general_inquiry", 0.90, "how do I use this?", "Here's how.")
    assert decision == "auto_handle"


def test_frustration_escalates():
    decision, reason = decide("software_update_issue", 0.90, "This is the worst!!", "Sorry to hear.")
    assert decision == "escalate"
    assert "frustration" in reason.lower()


# ── Draft template tests ────────────────────────────────────────────

def test_template_draft_all_intents():
    for intent in INTENTS:
        reply = draft_template("test message", intent, [])
        assert len(reply) > 20
        assert "DM" in reply or "dm" in reply.lower() or "help" in reply.lower()


# ── Retrieval tests ──────────────────────────────────────────────────

def test_retrieve_filters_by_intent():
    kb = [
        {"id": "1", "brand": "AppleSupport", "intent": "battery_or_power",
         "customer": "battery drains fast", "brand_reply": "DM us"},
        {"id": "2", "brand": "AppleSupport", "intent": "connectivity",
         "customer": "wifi broken", "brand_reply": "try resetting"},
    ]
    results = retrieve("my battery dies", "battery_or_power", kb, "AppleSupport")
    assert all(r["intent"] == "battery_or_power" for r in results)


def test_retrieve_excludes_self():
    kb = [
        {"id": "1", "brand": "AppleSupport", "intent": "battery_or_power",
         "customer": "my battery dies fast", "brand_reply": "DM us"},
    ]
    results = retrieve("my battery dies fast", "battery_or_power", kb, "AppleSupport")
    assert len(results) == 0  # Should exclude the exact match


# ── Judge & Infer integration tests ─────────────────────────────────

def test_judge_rule_based():
    from pipeline.judge import judge_rule_based
    res = judge_rule_based(
        customer="My wifi is not connecting",
        intent="connectivity",
        draft_reply="We'd like to help. Please DM us your device model.",
        evidence=[],
    )
    for dim in ["groundedness", "brand_voice", "actionability", "empathy", "safety"]:
        assert dim in res
        assert 1 <= res[dim] <= 5
    assert "overall_sendable" in res
    assert res["judge_type"] == "rule_based"


def test_infer_integration():
    from pipeline.core import infer
    kb = [
        {"id": "1", "brand": "AppleSupport", "intent": "battery_or_power",
         "customer": "battery drains fast", "brand_reply": "DM us your iOS version"},
    ]
    res = infer("My battery is dead", kb=kb, brand="AppleSupport", provider_override="heuristic")
    assert res["intent"] == "battery_or_power"
    assert res["decision"] in ("auto_handle", "escalate")
    assert "draft_reply" in res
    assert "retrieved_evidence" in res


def test_standard_battery_issue_auto_handles():
    decision, reason = decide("battery_or_power", 0.88, "My battery drains in 2 hours", "We'd like to help. Please DM us your device model.")
    assert decision == "auto_handle"


def test_standard_connectivity_issue_auto_handles():
    decision, reason = decide("connectivity", 0.88, "My wifi keeps dropping on my iPhone", "We'd love to help. Please DM us your iOS version.")
    assert decision == "auto_handle"


def test_requests_human_escalates():
    decision, reason = decide("general_inquiry", 0.90, "I want to speak to a human agent please", "Hi there.")
    assert decision == "escalate"
    assert "human" in reason.lower()


def test_adaptive_threshold_situations():
    from pipeline.core import calculate_adaptive_threshold

    # Routine accessory
    val, risk, _ = calculate_adaptive_threshold("general_inquiry", "my iphone case is broken")
    assert val == 0.55
    assert "low risk" in risk.lower()

    # Standard technical
    val, risk, _ = calculate_adaptive_threshold("battery_or_power", "my battery is draining fast")
    assert val == 0.68
    assert "medium risk" in risk.lower()

    # Sensitive / Security
    val, risk, _ = calculate_adaptive_threshold("account_or_security", "someone hacked my apple id password")
    assert val == 0.85
    assert "high risk" in risk.lower()

    # Elevated frustration
    val, risk, _ = calculate_adaptive_threshold("software_update_issue", "this is ridiculous and unacceptable")
    assert val == 0.80
    assert "frustration" in risk.lower() or "elevated" in risk.lower()


def test_routine_case_inquiry_auto_handles_at_lower_confidence():
    # 0.60 confidence would fail a rigid 0.72 slider, but safely auto-handles under adaptive 0.55 threshold
    decision, reason = decide(
        intent="general_inquiry",
        confidence=0.60,
        customer="my iphone case is broken where can I buy a replacement?",
        draft_text="We'd love to help with your case. You can explore cases at the Apple Store.",
        threshold="auto",
    )
    assert decision == "auto_handle"


def test_infer_returns_adaptive_threshold_metadata():
    from pipeline.core import infer
    kb = [
        {"id": "1", "brand": "AppleSupport", "intent": "general_inquiry",
         "customer": "broken case", "brand_reply": "Check Apple Store for cases"},
    ]
    res = infer("my iphone case is broken", kb=kb, brand="AppleSupport", provider_override="heuristic")
    assert "effective_threshold" in res
    assert res["effective_threshold"] == 0.55
    assert "risk_tier" in res
    assert "threshold_mode" in res
    assert res["threshold_mode"] == "auto"



