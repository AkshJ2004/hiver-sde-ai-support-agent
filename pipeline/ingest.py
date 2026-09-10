"""Create real customer-to-brand reply pairs from a TWCS-format CSV.

This module links inbound customer tweets to their outbound brand replies
using the response_tweet_id field. PII is scrubbed at ingestion time.

Usage:
    uv run -m pipeline.ingest --input data/raw/twcs.csv --brand AppleSupport --output data/processed/apple_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .provider import classify_heuristic
from .safety import scrub_pii


def build_pairs(frame: pd.DataFrame, brand: str | None = None, max_pairs: int = 5000) -> list[dict]:
    """Extract linked customer→brand reply pairs from a TWCS DataFrame.

    Args:
        frame: DataFrame with TWCS columns (tweet_id, author_id, inbound, text, response_tweet_id)
        brand: If set, only extract pairs where the brand reply is from this author_id
        max_pairs: Maximum number of pairs to extract
    """
    frame = frame.fillna("")
    by_id = {str(row.tweet_id): row for row in frame.itertuples()}
    pairs = []
    seen = set()

    for row in frame.itertuples():
        if len(pairs) >= max_pairs:
            break
        if str(row.inbound).lower() != "true" or not str(row.response_tweet_id):
            continue
        for reply_id in str(row.response_tweet_id).split(","):
            reply = by_id.get(reply_id.strip())
            if not reply or str(reply.inbound).lower() != "false":
                continue
            if brand and str(reply.author_id) != brand:
                continue

            customer_text = str(row.text).strip()
            brand_text = str(reply.text).strip()

            if len(customer_text) < 15 or len(brand_text) < 15:
                continue
            if customer_text in seen:
                continue
            seen.add(customer_text)

            customer = scrub_pii(customer_text)
            result = classify_heuristic(customer)

            pairs.append({
                "id": str(row.tweet_id),
                "brand": str(reply.author_id),
                "customer": customer,
                "brand_reply": scrub_pii(brand_text),
                "intent": result["intent"],
            })
            break
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest TWCS CSV into linked pairs")
    parser.add_argument("--input", required=True, help="Path to TWCS CSV")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--brand", default=None, help="Filter to a specific brand")
    parser.add_argument("--max-pairs", type=int, default=5000, help="Max pairs to extract")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    pairs = build_pairs(df, brand=args.brand, max_pairs=args.max_pairs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} linked customer/reply pairs")
