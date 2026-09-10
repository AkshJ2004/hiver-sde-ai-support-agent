# Decision Log

1. **Brand choice: AppleSupport** — The full TWCS dataset has AmazonHelp (170K replies) and AppleSupport (107K). AppleSupport was chosen because tech-support intents are more sharply defined than Amazon's mix of delivery/refund/product categories, making intent taxonomy design more tractable for a 6-class system.

2. **Six intents, not seven or more** — Keyword-frequency analysis showed six natural clusters covering >95% of messages. Adding finer-grained intents (e.g., splitting `software_update_issue` into iOS/macOS) would require more training data per class than the golden set supports.

3. **Copilot, not autonomous agent** — `auto_handle` means "eligible for human review and sending." No message is ever sent without a human in the loop. This is a deliberate safety constraint that simplifies trust arguments.

4. **Keyword classifier as the simple baseline, not a trained model** — With 200 labeled examples (auto-labeled at that), training a supervised classifier would overfit. The keyword heuristic is transparent and deterministic, making failure analysis tractable.

5. **Deterministic escalation policy, not learned** — Escalation rules (sensitive intents, low confidence, frustration, unverifiable claims) are hand-written policy, not predicted. This makes the safety boundary auditable and independent of classifier accuracy.

6. **PII scrubbing at ingestion, not at output** — Emails, phone numbers, and long digit strings are redacted before any processing. This prevents PII from appearing in KB, golden set, or artifacts even if downstream code has bugs.

7. **Token-overlap retrieval, not embeddings** — The dependency-free token-overlap scorer is the baseline. Embedding-based retrieval (e.g. Gemini text-embedding-004) is architecturally supported but not the default, to keep the project runnable offline.

8. **Template drafts as baseline, LLM RAG as full system** — Template drafts are deterministic and reproducible. LLM drafts require an API key (Google Gemini gemini-2.0-flash) but produce contextual, evidence-grounded replies. Both are evaluated side by side.

9. **Asymmetric cost: false auto-handle = 10× false escalate** — A dangerous message sent without review is far worse than a safe message unnecessarily queued. The 10:1 ratio is a policy choice reflecting brand risk tolerance.

10. **Rule-based judge as fallback, LLM judge as primary** — The project must be runnable without an API key. The rule-based judge checks for numeric hallucination, action words, and empathy keywords — it's transparent but cannot assess semantic quality. The LLM judge uses a 5-dimension rubric implemented via Gemini.

11. **Golden set auto-labeled, flagged for human review** — The 197 golden examples are auto-labeled by the heuristic classifier with `needs_review: true`. This is honest about the circular evaluation risk (Section 6 of the report). Manual correction is the single highest-impact next step.

12. **Stratified sampling with min-15-per-intent floor** — Proportional sampling would give `account_or_security` only 7 examples (3.6% of 200). The floor of 15 ensures every intent has enough examples for meaningful per-class metrics, at the cost of inflating rare-class representation.

13. **5K pairs as KB, not the full 107K** — Processing 107K pairs is slow and yields diminishing retrieval quality (token overlap degrades with corpus size). 5K gives sufficient within-intent coverage for the retrieval demo.

14. **Confidence threshold 0.72** — Set by inspection: keyword scores below 0.72 typically come from messages matching only 1 keyword, which is unreliable. Above 0.72 requires 2+ keyword matches, correlating with more confident classification.

15. **No Banking77 integration** — The optional secondary dataset has 77 fine-grained banking intents. These don't map to AppleSupport's domain and would require a separate intent-mapping layer. Noted for future multi-domain work.
