# Build Plan

## Completed

- Full TWCS dataset downloaded (2.8M tweets) and AppleSupport subset extracted (5K pairs)
- Data-driven 6-intent taxonomy from keyword-frequency analysis
- Three classification providers: majority (trivial), keyword (simple), LLM (full)
- RAG-grounded LLM draft generation with few-shot prompts
- Deterministic escalation policy with 5 safety gates
- 197-example golden evaluation set with stratified sampling and labeling guide
- Evaluation harness comparing 3 systems with intent/decision/judge metrics
- LLM-as-judge with 5-dimension rubric + rule-based fallback
- PII scrubbing at ingestion
- Streamlit interactive console with judge integration
- 15-entry decision log, full 6-section evaluation report
- Reproducible pipeline with clear README

## Remaining for production readiness

- Manually review and correct golden set labels (break circular evaluation)
- Connect Gemini API (GEMINI_API_KEY) for true LLM classification + RAG drafting evaluation
- Calibrate LLM judge against human scores (Cohen's kappa target: > 0.6)
- Embedding-based retrieval (e.g., text-embedding-004)
- Multi-turn context window for follow-up messages
- Expand to a second brand for generalization validation
