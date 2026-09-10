"""Export an inspectable, real-data retrieval bank from processed pairs."""
import argparse
import json
from pathlib import Path

def load_jsonl(path):
    with open(path, encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.pairs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf8")
    print(f"wrote {len(rows)} real customer/reply evidence rows")
