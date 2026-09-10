# Agent Handoff Quick Reference

> Full, comprehensive handoff documentation is maintained in [**`docs/handoff.md`**](docs/handoff.md).

## Quick Context
- **Brand & Domain**: AppleSupport AI Copilot on Twitter (TWCS dataset, 5K linked pairs in `data/processed/apple_pairs.jsonl`).
- **LLM Provider**: Google Gemini (`gemini-2.0-flash`, `google-generativeai`).
- **Tests**: 23/23 tests pass (`uv run pytest tests/ -v`).
- **Documentation**: All documentation is organized in [`docs/`](docs/).
- **Evaluation**: Run `uv run python scripts/run_eval.py` (offline or with `GEMINI_API_KEY`).
- **Interactive App**: Run `uv run streamlit run app.py`.

Please consult [**`docs/handoff.md`**](docs/handoff.md) for full module breakdown, commands, and pending tasks.
