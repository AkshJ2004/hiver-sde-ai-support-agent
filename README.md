# AppleSupport AI Copilot

This project is an AI-powered assistant designed to help customer support teams (specifically for Apple Support on Twitter) handle customer messages more effectively.

In simple terms, here is what the system does:
1. **Reads the Message:** It looks at what the customer is asking.
2. **Categorizes the Issue:** It figures out the main topic of the problem (like battery issues, software updates, etc.).
3. **Finds Past Solutions:** It searches for how similar problems were solved in the past.
4. **Drafts a Reply:** It writes a suggested response based on those past solutions.
5. **Makes a Decision:** It decides if the drafted reply is good enough to be sent to the customer (after a human checks it) or if the issue is too complex and needs to be escalated to a human expert.

This tool acts as an assistant or "copilot." It does not send messages automatically; it simply prepares the best possible draft so the support team can review and send it faster.

## Reproduce in under 15 minutes

### Prerequisites

- Python 3.13+ (or available to `uv`)
- [uv](https://docs.astral.sh/uv/) package manager
- ~600MB free disk for the TWCS dataset

### Quick Start (Start the App in 5 Seconds)

The processed dataset and evaluation sets are already bundled in this repository. You do **not** need to download Kaggle data or re-run pipeline scripts to use the app.

Simply run:
```powershell
uv run python -m uvicorn server:app --port 8000
```
*(Or double-click `run.bat` on Windows)*

Then open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### Optional: Full Data Pipeline Reproduction (From Raw Kaggle CSV)

If you wish to re-download the raw Twitter dataset and regenerate the pipeline artifacts from scratch:

```powershell
# 1. Download the TWCS dataset (requires Kaggle account)
uv run python -c "import kagglehub; print(kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'))"
# Copy twcs.csv from the printed path to data/raw/twcs.csv

# 2. Extract AppleSupport pairs (produces 5K linked conversations)
uv run python scripts/extract_apple.py

# 3. Build the golden evaluation set (197 stratified examples)
uv run python scripts/build_golden.py

# 4. Run the offline evaluation harness
uv run python scripts/run_eval.py
```

### With LLM (optional, requires Google Gemini API key)

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
$env:PROVIDER = "gemini"

# Re-run evaluation with Gemini LLM classification + RAG drafting
uv run python scripts/run_eval.py
```

## System Overview

```
Customer message
  → PII scrub (email, phone, digits)
  → Intent classification (keyword heuristic OR Gemini few-shot)
  → Retrieval (same-brand, same-intent, top-3 by token overlap)
  → Draft reply (template OR Gemini RAG with retrieved evidence)
  → Escalation decision (deterministic safety gates)
  → Output: intent, confidence, draft, decision, reason
```

### Three Evaluated Systems

| System | Classifier | Drafter | Decider |
|---|---|---|---|
| **Trivial** | Majority class | Fixed template | Always escalate |
| **Simple** | Keyword heuristic | Per-intent template | Rule-based gates |
| **Full** | Gemini few-shot | RAG + evidence | Rule-based gates |

## Intent Taxonomy

Derived from keyword-frequency analysis of 5,000 AppleSupport conversations:

`software_update_issue`, `battery_or_power`, `app_or_performance`,
`connectivity`, `account_or_security`, `general_inquiry`

## Key Results

| Metric | Trivial | Simple | Full |
|---|---|---|---|
| Accuracy | 22.3% | 100%* | 100%* |
| Macro-F1 | 6.1% | 100%* | 100%* |
| False auto-handles | 0 | 0 | 0 |
| Weighted cost | 109 | 43 | 43 |

*\*100% accuracy is circular — see "What is misleading" in the [report](docs/report.md).*

## Project Structure

```
├── docs/                 # All project documentation & guides
│   ├── architecture.md   # System architecture & component data flow
│   ├── decision_log.md   # 15 non-obvious design decisions & trade-offs
│   ├── goal.md           # Project scope and problem framing
│   ├── handoff.md        # Cross-agent state, context & task tracking
│   ├── labeling_guide.md # Stratified sampling methodology & labeling criteria
│   ├── plan.md           # Build plan & production roadmap
│   └── report.md         # Full 6-section evaluation report
├── scripts/              # Data extraction, golden set builder, evaluation harness
├── pipeline/             # Core modules (classify, retrieve, draft, decide, judge)
├── prompts/              # Versioned LLM prompts for Gemini (classify, draft, judge)
├── golden/               # 197-example evaluation set (golden_set.jsonl)
├── tests/                # Automated pytest suite (21 unit & regression tests)
├── artifacts/            # Generated outputs (eval_results.json, etc.)
├── data/                 # Raw & processed TWCS data (gitignored raw)
├── app.py                # Streamlit interactive test console
└── README.md             # Quick start and reproduction guide
```

## Documentation

All project documentation is centralized in [`docs/`](docs/):

- [Architecture & Data Flow](docs/architecture.md)
- [Decision Log (15 Key Decisions)](docs/decision_log.md)
- [Full Evaluation Report](docs/report.md)
- [Golden Set Labeling Guide](docs/labeling_guide.md)
- [Project Goals & Scope](docs/goal.md)
- [Implementation & Build Plan](docs/plan.md)
- [Agent Handoff & State](docs/handoff.md)

## Evaluation Caveats

- The golden set is auto-labeled by the keyword classifier; 100% accuracy is
  self-agreement, not quality. Manual label correction is the top next step.
- The rule-based judge checks for hallucinated numbers and keyword presence,
  not semantic quality. It is a transparent fallback.
- Stratified sampling inflates rare-class metrics vs. natural production mix.
- See [docs/report.md](docs/report.md) for full failure analysis and
  "what is misleading about my headline number."

## Dataset

Source: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
by ThoughtVector/Kaggle. Download it yourself and comply with its terms.
