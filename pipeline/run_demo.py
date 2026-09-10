"""Print a reproducible summary of the AppleSupport dataset."""
import json, sys, io
from collections import Counter
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parents[1]
PATH = ROOT / "data" / "processed" / "apple_pairs.jsonl"
if not PATH.exists():
    PATH = ROOT / "data" / "processed" / "real_pairs.jsonl"
if not PATH.exists():
    raise SystemExit("Run scripts/extract_apple.py first; see README.md")

rows = [json.loads(line) for line in PATH.read_text(encoding="utf8").splitlines() if line]
summary = {
    "source": "Customer Support on Twitter (thoughtvector/Kaggle)",
    "brand": "AppleSupport",
    "linked_customer_brand_pairs": len(rows),
    "brands": dict(Counter(row.get("brand", "unknown") for row in rows).most_common()),
    "intents": dict(Counter(row.get("intent", "unknown") for row in rows).most_common()),
}
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "artifacts" / "data_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf8"
)
print(json.dumps(summary, indent=2))
