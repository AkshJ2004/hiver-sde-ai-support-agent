# Agent Handoff & State Document

> **Last Updated**: Current Session (Adaptive Threshold & UI Streamlining)  
> **Status**: All 29 tests passing. Situational Adaptive Threshold active (calibrates dynamically from 0.55 for routine accessories to 0.85 for security). Custom input interface streamlined with quick test scenario chips. All Twitter/X branding removed.  
> **Primary Provider**: Google Gemini (`gemini-flash-latest`, `google-generativeai`) with offline fallback to heuristic classifier & rule-based judge.

---

## 1. Executive Summary & Purpose

This repository implements an enterprise AI support copilot for **AppleSupport**.
The system:
1. Classifies incoming customer messages into 6 data-derived intents.
2. Retrieves historically verified same-brand resolutions using token overlap.
3. Drafts grounded replies via Google Gemini RAG (or deterministic templates).
4. Enforces a **Situational Adaptive Escalation Policy** (`auto_handle` vs `escalate`) with stated reasons:
   - **Low Risk (0.55)**: Routine inquiries, accessories, cases, chargers, store hours, retail questions — auto-handled smoothly without unnecessary human friction.
   - **Medium Risk (0.68)**: Standard technical troubleshooting (battery, Wi-Fi, app crashes, software updates).
   - **High Risk (0.85)**: Sensitive account/security issues, Apple ID lockouts, password resets, billing disputes — strictly guarded.
   - **Elevated Risk (0.80)**: Heightened customer frustration or urgency.
   - **Hard Gates**: Explicit human agent requests and unauthorized financial/hardware commitments always escalate.
5. Evaluates 3 systems side-by-side (Trivial, Simple, Full) using intent metrics, asymmetric decision cost, and LLM-as-judge reply quality scores.

---

## 2. Directory Structure & File Index

All markdown documentation is strictly housed in `docs/`:

```
Protfolio/
├── docs/                     # All project documentation
│   ├── architecture.md       # Architecture diagram, data flow & component table
│   ├── decision_log.md       # 15 non-obvious engineering & design decisions
│   ├── goal.md               # Project framing, problem statement & copilot philosophy
│   ├── handoff.md            # THIS FILE: Cross-agent state, context & task roadmap
│   ├── labeling_guide.md     # Golden set stratified sampling & human labeling rubric
│   ├── plan.md               # Completed build phases & production roadmap
│   └── report.md             # Full 6-section evaluation report with failure analysis
├── web/                      # HTML5, Vanilla CSS & Modern JS Web UI
│   ├── index.html            # Apple glassmorphic copilot interface
│   ├── style.css             # Responsive design system & CSS tokens
│   └── app.js                # Async client-side controller & live visualizer
├── server.py                 # FastAPI backend serving static UI & REST endpoints
├── scripts/
│   ├── extract_apple.py      # Extracts 5,000 linked AppleSupport pairs from twcs.csv
│   ├── build_golden.py       # Stratified sampling builder for 197-example evaluation set
│   ├── run_eval.py           # 3-system evaluation harness with LLM/rule-based judge
│   └── explore_data.py       # Dataset inspection and frequency analysis
├── pipeline/
│   ├── __init__.py           # Package init
│   ├── constants.py          # 6-intent taxonomy, sensitive intents, threshold (0.72)
│   ├── safety.py             # PII scrubbing regex, frustration detection, claim verification
│   ├── provider.py           # Gemini 3.6 Flash classifier + heuristic & majority baselines
│   ├── core.py               # RAG retrieval, Gemini/template drafting, escalation decider, infer()
│   ├── judge.py              # Gemini 3.6 Flash judge (5 dimensions) + rule-based fallback
│   ├── evaluate.py           # Precision, recall, F1, cost matrix metrics
│   ├── ingest.py             # Pair-linking logic & PII scrubber for TWCS CSV
│   └── run_demo.py           # CLI summary printer
├── prompts/                  # Versioned prompt templates
│   ├── classify_v2.txt       # Few-shot Gemini intent classification prompt
│   ├── draft_v2.txt          # Gemini grounded RAG reply generation prompt
│   └── judge_v2.txt          # 5-dimension rubric Gemini judge prompt
├── golden/
│   └── golden_set.jsonl      # 197 stratified evaluation examples
├── tests/
│   └── test_pipeline.py      # 23 automated tests (pytest)
├── artifacts/
│   └── eval_results.json     # Saved JSON results of last run_eval.py execution
├── data/
│   ├── raw/                  # Raw TWCS twcs.csv (gitignored)
│   └── processed/            # apple_pairs.jsonl (5,000 linked pairs)
├── pyproject.toml            # Project configuration & dependencies
├── requirements.txt          # Exported dependency list
├── .env                      # Active environment configuration
├── .env.example              # Environment variables template
├── handoff.md                # Root reference pointing to docs/handoff.md
└── README.md                 # 15-minute reproduction guide & headline summary
```

---

## 3. Environment & Running Instructions

### Dependencies & Virtual Environment
The project uses `uv` with Python 3.13:
- Core dependencies: `pandas>=2.0`, `scikit-learn>=1.3`, `streamlit>=1.40`, `kagglehub>=1.0`, `google-generativeai>=0.8`, `python-dotenv>=1.0`
- Dev dependencies: `pytest`
- Supports automatic `.env` loading from root.

### Essential Commands
```powershell
# 1. Run all unit & regression tests
uv run pytest tests/ -v

# 2. Run evaluation harness (offline mode: trivial, simple, and heuristic full)
$env:PYTHONIOENCODING = "utf-8"; uv run python scripts/run_eval.py

# 3. Run evaluation harness with Google Gemini
$env:GEMINI_API_KEY = "your-api-key"
$env:PROVIDER = "gemini"
uv run python scripts/run_eval.py

# 4. Run Streamlit interactive test console
uv run streamlit run app.py

# 5. Re-generate golden set if needed
uv run python scripts/build_golden.py
```

---

## 4. Current State & Verification Results

- **Automated Tests**: 23/23 passing (`tests/test_pipeline.py`).
  - Covers: keyword classification across intents, majority classifier, empty default, PII scrubbing (emails, phones, long digit strings), frustration detection, order reference check, sensitive intent auto-escalation, low confidence escalation, high confidence auto-handle, template drafting, retrieval intent filtering & self-exclusion, rule-based judge dimensions, and end-to-end `infer()` pipeline.
- **Evaluation Harness**: Verified working end-to-end. Produces `artifacts/eval_results.json`.
- **Streamlit App**: Configured with live provider selection (`heuristic`, `gemini`, `majority`), confidence slider, real-time message inspector, and error fallbacks.

---

## 5. Critical Domain Nuances & Evaluation Truths

Any incoming agent must understand these core design decisions (detailed in `docs/decision_log.md` and `docs/report.md`):

1. **Copilot, NOT Autonomous Bot**:
   `auto_handle` means *eligible for human one-click sending*. No message is ever dispatched to Twitter/X without agent oversight.
2. **Circular Accuracy Caveat (Must be highlighted to reviewers)**:
   The current 100% accuracy on Simple and Full baselines is **circular self-agreement** because `scripts/build_golden.py` used the heuristic classifier to auto-label the 197 golden set examples. True evaluation requires human label review (`golden_set.jsonl` contains `"needs_review": true` flags).
3. **Asymmetric Safety Cost**:
   False auto-handle cost = 10, False escalate cost = 1. A bad tweet sent to a customer is 10× worse than an unnecessary human review ticket.
4. **Data-Driven 6-Intent Taxonomy**:
   `software_update_issue`, `battery_or_power`, `app_or_performance`, `connectivity`, `account_or_security`, `general_inquiry`. `account_or_security` is hard-coded as **SENSITIVE** and always escalated.

---

## 6. What Next Agent Needs to Do

When resuming or continuing with this codebase:

1. **If User provides `GEMINI_API_KEY`**:
   - Run:
     ```powershell
     $env:GEMINI_API_KEY = "..."
     $env:PROVIDER = "gemini"
     uv run python scripts/run_eval.py
     ```
   - Inspect `artifacts/eval_results.json` to verify Gemini classification and LLM judge scores.
2. **Golden Set Human Review**:
   - Review examples in `golden/golden_set.jsonl` with `"needs_review": true`.
   - Update ground truth labels (`intent` and `gold_decision`), flip `"needs_review": false`.
   - Re-run eval to observe true non-circular classifier divergence.
3. **Judge Calibration**:
   - Compare ~30 judge evaluations against human review scores to calculate inter-rater agreement (Cohen's Kappa).
4. **Maintenance**:
   - If any new files are created, ensure docs stay in `docs/` and update this `handoff.md` after every change.
