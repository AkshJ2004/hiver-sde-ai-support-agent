"""Build the golden evaluation set via stratified sampling.

Samples ~200 examples from the AppleSupport pairs, stratified by
keyword-inferred intent, then auto-labels intents using the heuristic
classifier. The user MUST review and correct labels manually.

Usage:
  uv run python scripts/build_golden.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.provider import classify_heuristic
from pipeline.safety import high_frustration

PAIRS_PATH = ROOT / "data" / "processed" / "apple_pairs.jsonl"
GOLDEN_PATH = ROOT / "golden" / "golden_set.jsonl"
TARGET_TOTAL = 200

random.seed(42)  # Reproducible sampling


def load_pairs():
    with open(PAIRS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_sample(pairs: list[dict], target: int) -> list[dict]:
    """Stratified sampling: proportional to intent frequency, with
    a minimum of 15 per intent to ensure coverage of rare classes."""

    # Classify all pairs
    by_intent = defaultdict(list)
    for p in pairs:
        result = classify_heuristic(p["customer"])
        p["_inferred_intent"] = result["intent"]
        p["_inferred_confidence"] = result["confidence"]
        by_intent[result["intent"]].append(p)

    print("Intent distribution in full dataset:")
    for intent, items in sorted(by_intent.items(), key=lambda x: -len(x[1])):
        print(f"  {intent}: {len(items)}")

    # Allocate samples: min 15 per intent, rest proportional
    n_intents = len(by_intent)
    min_per = 15
    guaranteed = min_per * n_intents
    remaining = max(0, target - guaranteed)

    total = sum(len(v) for v in by_intent.values())
    sampled = []

    for intent, items in by_intent.items():
        # Proportional allocation of the remaining budget
        proportion = len(items) / total
        n = min_per + int(remaining * proportion)
        n = min(n, len(items))  # Can't sample more than we have

        # Prioritize diversity: mix of high/low confidence, frustrated/calm
        random.shuffle(items)
        # Sort to get a mix: some high-confidence, some low, some frustrated
        items_sorted = sorted(items, key=lambda x: (
            x["_inferred_confidence"],
            high_frustration(x["customer"]),
        ))
        # Take evenly spaced samples for diversity
        step = max(1, len(items_sorted) // n)
        selected = items_sorted[::step][:n]
        sampled.extend(selected)

    random.shuffle(sampled)
    return sampled[:target]


def build_golden(pairs: list[dict]) -> list[dict]:
    """Build golden set with auto-labels for human review."""
    sampled = stratified_sample(pairs, TARGET_TOTAL)

    golden = []
    for p in sampled:
        result = classify_heuristic(p["customer"])

        # Auto-determine gold decision based on safety rules
        from pipeline.constants import SENSITIVE, ACTIONABLE, CONF_THRESHOLD
        from pipeline.safety import has_order_reference

        intent = result["intent"]
        confidence = result["confidence"]

        if intent in SENSITIVE:
            gold_decision = "escalate"
        elif confidence < CONF_THRESHOLD:
            gold_decision = "escalate"
        elif high_frustration(p["customer"]):
            gold_decision = "escalate"
        else:
            gold_decision = "auto_handle"

        golden.append({
            "id": p["id"],
            "customer": p["customer"],
            "brand_reply_actual": p["brand_reply"],
            "intent": intent,
            "gold_decision": gold_decision,
            "confidence": confidence,
            "needs_review": True,  # Flag: human must verify this label
            "notes": "",
        })

    return golden


def main():
    if not PAIRS_PATH.exists():
        print(f"ERROR: {PAIRS_PATH} not found. Run scripts/extract_apple.py first.")
        sys.exit(1)

    pairs = load_pairs()
    print(f"Loaded {len(pairs)} pairs")

    golden = build_golden(pairs)

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        for row in golden:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(golden)} golden examples to {GOLDEN_PATH}")

    # Print distribution
    intent_counts = Counter(r["intent"] for r in golden)
    decision_counts = Counter(r["gold_decision"] for r in golden)
    print(f"\nIntent distribution in golden set:")
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count}")
    print(f"\nDecision distribution:")
    for decision, count in decision_counts.most_common():
        print(f"  {decision}: {count}")

    print(f"\n[!] IMPORTANT: You must manually review and correct labels in {GOLDEN_PATH}")
    print("    Each row has 'needs_review': true — set to false after verification.")


if __name__ == "__main__":
    main()
