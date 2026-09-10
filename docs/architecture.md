# Architecture

```
TWCS CSV (2.8M rows)
  → Filter for AppleSupport (107K outbound)
  → Link customer→brand pairs (5K)
  → PII scrubbing (email, phone, digits)
  → Keyword-based intent classification (6 intents)
  → Token-overlap retrieval (same-brand, same-intent, top-3)
  → Draft reply (template baseline OR LLM RAG)
  → Deterministic escalation gates
  → Web UI (HTML5, Vanilla CSS, JS) via FastAPI
```

## Components

| Module | Purpose |
|---|---|
| `scripts/extract_apple.py` | Extract AppleSupport pairs from full TWCS |
| `scripts/build_golden.py` | Stratified sampling for golden evaluation set |
| `scripts/run_eval.py` | Evaluation harness: 3 systems × golden set |
| `pipeline/provider.py` | Intent classification (majority / keyword / Gemini 3.6 Flash) |
| `pipeline/core.py` | Retrieve → draft → decide pipeline |
| `pipeline/judge.py` | Reply quality scoring (rule-based / Gemini 3.6 Flash) |
| `pipeline/safety.py` | PII scrubbing, frustration detection, claim verification |
| `pipeline/evaluate.py` | Reusable metric functions |
| `pipeline/constants.py` | Shared intent taxonomy and policy thresholds |
| `server.py` | FastAPI backend serving REST API & static assets |
| `web/` | Modern Apple-inspired UI (HTML5, Vanilla CSS, JS) |

## Escalation Policy

Always escalate: `account_or_security` intents, confidence < 0.72,
high frustration (profanity/threats), unverifiable numeric claims,
actionable intents without a device/version reference.

## Data Flow

Golden set → `run_eval.py` → runs trivial, simple, full systems →
computes intent/decision/judge metrics → `artifacts/eval_results.json`
