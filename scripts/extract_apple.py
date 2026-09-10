"""Extract AppleSupport customer→brand pairs from the full TWCS dataset.

Outputs:
  data/processed/apple_pairs.jsonl  — all linked pairs (used as KB)
  artifacts/data_summary.json       — updated stats

Usage:
  uv run python scripts/extract_apple.py
"""
import json, sys, io
from collections import Counter
from pathlib import Path

import pandas as pd

# Fix Windows encoding for emoji/Unicode in print
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.safety import scrub_pii

BRAND = "AppleSupport"
MAX_PAIRS = 5000  # Enough for KB + golden set; keeps processing fast

print(f"Loading full TWCS dataset...")
df = pd.read_csv(ROOT / "data" / "raw" / "twcs.csv")
print(f"Total rows: {len(df):,}")

df = df.fillna("")

# Build tweet_id -> row lookup
by_id = {}
for row in df.itertuples():
    by_id[str(row.tweet_id)] = row

# Extract linked customer→brand pairs for AppleSupport
pairs = []
seen_customers = set()

for row in df.itertuples():
    if len(pairs) >= MAX_PAIRS:
        break
    # Only inbound (customer) tweets
    if str(row.inbound).lower() != "true":
        continue
    if not str(row.response_tweet_id):
        continue

    for reply_id in str(row.response_tweet_id).split(","):
        reply = by_id.get(reply_id.strip())
        if not reply:
            continue
        if str(reply.author_id) != BRAND:
            continue
        if str(reply.inbound).lower() != "false":
            continue

        customer_text = str(row.text).strip()
        brand_text = str(reply.text).strip()

        # Skip very short or empty
        if len(customer_text) < 15 or len(brand_text) < 15:
            continue
        # Skip near-duplicates
        if customer_text in seen_customers:
            continue
        seen_customers.add(customer_text)

        pairs.append({
            "id": str(row.tweet_id),
            "brand": BRAND,
            "customer": scrub_pii(customer_text),
            "brand_reply": scrub_pii(brand_text),
            "created_at": str(row.created_at),
        })
        break  # Take first reply only

print(f"Extracted {len(pairs):,} linked AppleSupport customer→brand pairs")

# Write pairs
out_path = ROOT / "data" / "processed" / "apple_pairs.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Wrote {out_path}")

# Analyze content for intent taxonomy discovery
print(f"\n--- Content analysis for intent taxonomy ---")

# Keyword frequency analysis
keyword_groups = {
    "battery/power": ["battery", "charge", "charging", "power", "drain", "dies"],
    "update/ios": ["update", "ios", "upgrade", "ios11", "ios12", "software"],
    "app_crash/freeze": ["crash", "freeze", "freezing", "stuck", "hang", "lag", "slow", "frozen"],
    "wifi/connectivity": ["wifi", "wi-fi", "bluetooth", "cellular", "signal", "connection", "connect"],
    "storage/memory": ["storage", "memory", "space", "full", "icloud", "backup"],
    "screen/display": ["screen", "display", "touch", "touchscreen", "broken screen"],
    "account/apple_id": ["apple id", "appleid", "password", "login", "sign in", "locked out", "account", "two factor", "2fa"],
    "purchase/billing": ["refund", "charged", "charge", "billing", "purchase", "subscription", "payment", "money"],
    "delivery/order": ["delivery", "order", "shipped", "shipping", "tracking", "arrived"],
    "general_how_to": ["how do i", "how to", "how can", "where is", "can i", "is it possible"],
    "frustration/complaint": ["worst", "terrible", "hate", "angry", "furious", "unacceptable", "disappointed", "fix this", "broken"],
}

group_counts = {}
for group, keywords in keyword_groups.items():
    count = sum(1 for p in pairs if any(k in p["customer"].lower() for k in keywords))
    group_counts[group] = count

print("\nKeyword group hits (not mutually exclusive):")
for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
    print(f"  {group}: {count} ({100*count/len(pairs):.1f}%)")

# Summary stats
summary = {
    "dataset": "Customer Support on Twitter (thoughtvector/Kaggle)",
    "brand": BRAND,
    "total_twcs_rows": len(df),
    "extracted_pairs": len(pairs),
    "avg_customer_length": round(sum(len(p["customer"]) for p in pairs) / len(pairs), 1),
    "avg_reply_length": round(sum(len(p["brand_reply"]) for p in pairs) / len(pairs), 1),
    "keyword_group_hits": group_counts,
}
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "artifacts" / "data_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nWrote artifacts/data_summary.json")
