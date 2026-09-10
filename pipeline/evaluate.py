"""Evaluation metrics for the AppleSupport copilot.

This module provides reusable metric functions called by scripts/run_eval.py.
It does NOT run standalone — use run_eval.py as the entry point.
"""
from __future__ import annotations

from collections import Counter
from .constants import INTENTS


def intent_metrics(gold: list[dict], predictions: list[dict]) -> dict:
    """Compute intent classification accuracy, macro-F1, per-intent P/R/F1."""
    n = len(gold)
    if not n:
        return {"n": 0, "accuracy": 0, "macro_f1": 0, "per_intent": {}}

    correct = sum(g["intent"] == p["intent"] for g, p in zip(gold, predictions))

    per_intent = {}
    for label in INTENTS:
        tp = sum(g["intent"] == label and p["intent"] == label for g, p in zip(gold, predictions))
        fp = sum(g["intent"] != label and p["intent"] == label for g, p in zip(gold, predictions))
        fn = sum(g["intent"] == label and p["intent"] != label for g, p in zip(gold, predictions))

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        per_intent[label] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": sum(g["intent"] == label for g in gold),
        }

    macro_f1 = sum(x["f1"] for x in per_intent.values()) / len(per_intent)
    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "macro_f1": round(macro_f1, 3),
        "per_intent": per_intent,
    }


def decision_metrics(gold: list[dict], predictions: list[dict]) -> dict:
    """Compute false-auto-handle/escalate rates and weighted cost."""
    n = len(gold)
    if not n:
        return {}

    false_auto = sum(
        g.get("gold_decision") == "escalate" and p.get("decision") == "auto_handle"
        for g, p in zip(gold, predictions)
    )
    false_esc = sum(
        g.get("gold_decision") == "auto_handle" and p.get("decision") == "escalate"
        for g, p in zip(gold, predictions)
    )
    auto_rate = sum(p.get("decision") == "auto_handle" for p in predictions) / n

    return {
        "false_auto_handles": false_auto,
        "false_escalates": false_esc,
        "weighted_cost": false_auto * 10 + false_esc,
        "auto_handle_rate": round(auto_rate, 3),
        "decision_accuracy": round(
            sum(g.get("gold_decision") == p.get("decision") for g, p in zip(gold, predictions)) / n,
            3,
        ),
    }
