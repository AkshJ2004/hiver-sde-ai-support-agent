"""Evaluation harness: runs three systems on the golden set and reports metrics.

Systems compared:
  1. Trivial baseline  — majority class + always-escalate
  2. Simple baseline   — keyword classifier + template draft + rule gates
  3. Full system       — Gemini LLM classifier + RAG draft + rule gates (if API key set)

Metrics reported:
  - Intent classification: accuracy, macro-F1, per-intent P/R/F1, confusion matrix
  - Decision quality: false-auto-handle rate, false-escalate rate, weighted cost
  - Reply quality: LLM-as-judge scores (if API key) or rule-based scores

Usage:
  uv run python scripts/run_eval.py
  PROVIDER=gemini GEMINI_API_KEY=... uv run python scripts/run_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.constants import CONF_THRESHOLD, INTENTS, SENSITIVE
from pipeline.core import decide, draft_template, infer, retrieve
from pipeline.judge import judge
from pipeline.provider import classify_heuristic, classify_majority

GOLDEN_PATH = ROOT / "golden" / "golden_set.jsonl"
KB_PATH = ROOT / "data" / "processed" / "apple_pairs.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "eval_results.json"

BRAND = "AppleSupport"

# False-auto-handle costs 10x more than false-escalate (safety asymmetry)
COST_FALSE_AUTO = 10
COST_FALSE_ESC = 1


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_intent_metrics(gold: list[dict], predictions: list[dict]) -> dict:
    """Compute classification accuracy, macro-F1, and per-intent P/R/F1."""
    n = len(gold)
    correct = sum(g["intent"] == p["intent"] for g, p in zip(gold, predictions))

    per_intent = {}
    for label in INTENTS:
        tp = sum(g["intent"] == label and p["intent"] == label for g, p in zip(gold, predictions))
        fp = sum(g["intent"] != label and p["intent"] == label for g, p in zip(gold, predictions))
        fn = sum(g["intent"] == label and p["intent"] != label for g, p in zip(gold, predictions))

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        per_intent[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(g["intent"] == label for g in gold),
        }

    macro_f1 = sum(x["f1"] for x in per_intent.values()) / len(per_intent)

    # Confusion matrix
    intent_list = list(INTENTS)
    confusion = [[0] * len(intent_list) for _ in intent_list]
    for g, p in zip(gold, predictions):
        if g["intent"] in intent_list and p["intent"] in intent_list:
            gi = intent_list.index(g["intent"])
            pi = intent_list.index(p["intent"])
            confusion[gi][pi] += 1

    return {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0,
        "macro_f1": round(macro_f1, 3),
        "per_intent": per_intent,
        "confusion_matrix": {"labels": intent_list, "matrix": confusion},
    }


def compute_decision_metrics(gold: list[dict], predictions: list[dict]) -> dict:
    """Compute false-auto-handle, false-escalate, weighted cost."""
    n = len(gold)
    false_auto = sum(
        g["gold_decision"] == "escalate" and p["decision"] == "auto_handle"
        for g, p in zip(gold, predictions)
    )
    false_esc = sum(
        g["gold_decision"] == "auto_handle" and p["decision"] == "escalate"
        for g, p in zip(gold, predictions)
    )
    auto_rate = sum(p["decision"] == "auto_handle" for p in predictions) / n if n else 0

    return {
        "false_auto_handles": false_auto,
        "false_escalates": false_esc,
        "weighted_cost": false_auto * COST_FALSE_AUTO + false_esc * COST_FALSE_ESC,
        "auto_handle_rate": round(auto_rate, 3),
        "decision_accuracy": round(
            sum(g["gold_decision"] == p["decision"] for g, p in zip(gold, predictions)) / n, 3
        ) if n else 0,
    }


def run_trivial_baseline(golden: list[dict], kb: list[dict]) -> list[dict]:
    """Trivial: majority class, always escalate."""
    results = []
    for g in golden:
        cls = classify_majority(g["customer"])
        results.append({
            "intent": cls["intent"],
            "confidence": cls["confidence"],
            "decision": "escalate",
            "decision_reason": "trivial baseline: always escalate",
            "draft_reply": "Thank you for reaching out. A support agent will assist you shortly.",
        })
    return results


def run_simple_baseline(golden: list[dict], kb: list[dict]) -> list[dict]:
    """Simple: keyword classifier + template draft + rule gates."""
    results = []
    for g in golden:
        cls = classify_heuristic(g["customer"])
        evidence = retrieve(g["customer"], cls["intent"], kb, BRAND)
        reply = draft_template(g["customer"], cls["intent"], evidence)
        decision, reason = decide(cls["intent"], cls["confidence"], g["customer"], reply)
        results.append({
            "intent": cls["intent"],
            "confidence": cls["confidence"],
            "decision": decision,
            "decision_reason": reason,
            "draft_reply": reply,
            "retrieved_evidence": evidence[:3],
        })
    return results


def run_full_system(golden: list[dict], kb: list[dict]) -> list[dict]:
    """Full: Gemini LLM classifier + RAG draft (or heuristic if no API key)."""
    has_api = bool(os.getenv("GEMINI_API_KEY"))
    provider = "gemini" if has_api else "heuristic"
    print(f"  Full system using provider: {provider}")

    results = []
    for i, g in enumerate(golden):
        result = infer(g["customer"], kb, BRAND, provider_override=provider)
        results.append(result)
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(golden)}...")
    return results


def run_judge_on_results(
    golden: list[dict], results: list[dict], max_judge: int = 50
) -> list[dict]:
    """Run the judge on a subset of results."""
    judge_results = []
    for i, (g, r) in enumerate(zip(golden[:max_judge], results[:max_judge])):
        evidence = r.get("retrieved_evidence", [])
        score = judge(g["customer"], r["intent"], r.get("draft_reply", ""), evidence)
        judge_results.append(score)
        if (i + 1) % 10 == 0:
            print(f"  Judged {i + 1}/{min(max_judge, len(golden))}...")
    return judge_results


def aggregate_judge_scores(scores: list[dict]) -> dict:
    """Compute average judge scores across all judged examples."""
    if not scores:
        return {}

    dims = ["groundedness", "brand_voice", "actionability", "empathy", "safety"]
    avgs = {}
    for dim in dims:
        vals = [s.get(dim, 0) for s in scores if dim in s]
        avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 0

    sendable = sum(1 for s in scores if s.get("overall_sendable")) / len(scores)
    avgs["overall_sendable_rate"] = round(sendable, 3)
    avgs["n_judged"] = len(scores)
    avgs["judge_type"] = scores[0].get("judge_type", "unknown") if scores else "unknown"
    return avgs


def main():
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found. Run scripts/build_golden.py first.")
        sys.exit(1)
    if not KB_PATH.exists():
        print(f"ERROR: {KB_PATH} not found. Run scripts/extract_apple.py first.")
        sys.exit(1)

    golden = load_jsonl(GOLDEN_PATH)
    kb = load_jsonl(KB_PATH)
    print(f"Loaded {len(golden)} golden examples, {len(kb)} KB entries")

    # ── Run all three systems ────────────────────────────────────────
    print("\n=== Trivial Baseline ===")
    trivial_preds = run_trivial_baseline(golden, kb)

    print("\n=== Simple Baseline (keyword + template) ===")
    simple_preds = run_simple_baseline(golden, kb)

    print("\n=== Full System ===")
    full_preds = run_full_system(golden, kb)

    # ── Compute metrics ──────────────────────────────────────────────
    results = {}
    for name, preds in [("trivial", trivial_preds), ("simple", simple_preds), ("full", full_preds)]:
        print(f"\n--- Metrics: {name} ---")
        intent_m = compute_intent_metrics(golden, preds)
        decision_m = compute_decision_metrics(golden, preds)
        print(f"  Accuracy: {intent_m['accuracy']}, Macro-F1: {intent_m['macro_f1']}")
        print(f"  False auto-handles: {decision_m['false_auto_handles']}, "
              f"False escalates: {decision_m['false_escalates']}, "
              f"Weighted cost: {decision_m['weighted_cost']}")
        results[name] = {"intent_metrics": intent_m, "decision_metrics": decision_m}

    # ── Run judge on outputs ─────────────────────────────────────────
    print("\n=== Running judge on full system drafts ===")
    judge_scores = run_judge_on_results(golden, full_preds, max_judge=50)
    results["full"]["judge_scores"] = aggregate_judge_scores(judge_scores)
    results["full"]["judge_details"] = judge_scores

    print("\n=== Running judge on simple baseline drafts ===")
    simple_judge = run_judge_on_results(golden, simple_preds, max_judge=50)
    results["simple"]["judge_scores"] = aggregate_judge_scores(simple_judge)

    print(f"\nFull system judge averages: {results['full']['judge_scores']}")
    print(f"Simple baseline judge averages: {results['simple']['judge_scores']}")

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
