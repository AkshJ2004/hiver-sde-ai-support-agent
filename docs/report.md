# AppleSupport AI Copilot — Evaluation Report

## 1. Problem Framing

### What "good" means for AppleSupport

AppleSupport handles ~107K conversations in the TWCS dataset, predominantly
about iOS update regressions, battery issues, app crashes, and connectivity.
A "good" agent must:

- **Correctly classify** the customer's primary intent into one of six categories
- **Draft a reply** that matches AppleSupport's real tone: empathetic, slightly formal, action-oriented (typically directing to DM)
- **Never fabricate facts** — no hallucinated order numbers, claimed actions, or invented technical steps
- **Safely decide** whether to auto-handle or escalate, erring on the side of caution

### What we chose NOT to build

- **Multi-turn conversation handling**: We process single customer turns, not full threads. Real support requires context accumulation.
- **Sentiment scoring**: We use frustration keywords as escalation triggers, not a trained sentiment model.
- **CRM integration**: No real account lookup, order verification, or ticket creation.
- **Proactive outreach**: The agent is reactive only.
- **Automated sending**: This is a *copilot* — `auto_handle` means "eligible for human review and sending", not that messages are sent automatically.

## 2. System Architecture

```
Customer message
  → PII scrub (regex: email, phone, long digits)
  → Intent classification (keyword heuristic OR LLM few-shot)
  → Retrieval (same-brand, same-intent, token-overlap scoring)
  → Reply drafting (template OR LLM RAG with retrieved evidence)
  → Escalation decision (deterministic policy: sensitive intents,
     low confidence, frustration, unverifiable claims)
  → Output: intent, confidence, draft, decision + reason
```

Three system configurations are evaluated:

| System | Classifier | Drafter | Decider |
|---|---|---|---|
| Trivial baseline | Majority class (`software_update_issue`) | Fixed template | Always escalate |
| Simple baseline | Keyword heuristic (14-word lists per intent) | Per-intent template | Rule-based gates |
| Full system | LLM few-shot (GPT-4o-mini) | LLM RAG + retrieved evidence | Rule-based gates |

## 3. Intent Taxonomy

Derived from keyword-frequency analysis of 5,000 AppleSupport conversations:

| Intent | Examples | Frequency |
|---|---|---|
| software_update_issue | "Since iOS 11 my phone is slow" | 26.5% |
| battery_or_power | "Battery dies in 3 hours" | 9.0% |
| app_or_performance | "Apps keep crashing and freezing" | 10.0% |
| connectivity | "WiFi keeps disconnecting" | 5.3% |
| account_or_security | "Locked out of my Apple ID" | 3.6% |
| general_inquiry | "How do I transfer photos?" | 45.6% |

## 4. Results vs. Baselines

### Intent Classification (on 197 golden examples)

| Metric | Trivial | Simple | Full (heuristic*) |
|---|---|---|---|
| Accuracy | 0.223 | 1.000* | 1.000* |
| Macro-F1 | 0.061 | 1.000* | 1.000* |

*\*The simple and full systems show 1.0 accuracy because the golden set was auto-labeled by the same keyword heuristic. This is a circular evaluation — see Section 6. With an LLM classifier and manually corrected labels, these numbers will differ.*

### Decision Quality

| Metric | Trivial | Simple | Full |
|---|---|---|---|
| False auto-handles | 0 | 0 | 0 |
| False escalates | 109 | 43 | 43 |
| Weighted cost (10×FA + FE) | 109 | 43 | 43 |
| Auto-handle rate | 0% | 55.3% | 55.3% |

The trivial baseline's zero false-auto-handles is trivially achieved by always escalating. The simple/full systems achieve the same safety (0 false auto-handles) while resolving 55% of messages without escalation.

### Reply Quality (rule-based judge, 50 examples)

| Dimension | Simple | Full |
|---|---|---|
| Groundedness | 4.0 / 5 | 4.0 / 5 |
| Brand Voice | 3.0 / 5 | 3.0 / 5 |
| Actionability | 4.0 / 5 | 4.0 / 5 |
| Empathy | 3.1 / 5 | 3.1 / 5 |
| Safety | 4.0 / 5 | 4.0 / 5 |
| Sendable rate | 100% | 100% |

*Note: Both systems use template drafts without an API key. With LLM drafting enabled, the full system produces personalized, evidence-grounded replies that should score higher on brand voice and empathy. The rule-based judge is a transparent lower bound — it cannot assess semantic quality.*

## 5. Failure Analysis — Top 5 Failure Modes

### 1. Multi-intent messages misclassified
**Example**: *"Ever since the update my wifi keeps dropping and apps crash"*
The keyword classifier sees "update" (software_update_issue), "wifi" (connectivity), AND "crash" (app_or_performance) and picks whichever has the highest keyword count, ignoring the primary complaint.

**Hypothesis**: The bag-of-words approach cannot weigh context. An LLM with few-shot examples handles multi-intent messages by identifying the root cause.

### 2. Sarcasm and frustration misread as calm
**Example**: *"Love the new update @AppleSupport!!!! My phone is basically a brick now"*
The frustration detector looks for keywords like "hate", "furious". Sarcastic praise ("love") with exclamation marks is missed.

**Hypothesis**: The regex-based frustration detector needs irony-aware features. An LLM judge can detect sarcasm, but the keyword classifier cannot.

### 3. Follow-up messages lacking original context
**Example**: *"@AppleSupport I have the latest version iOS. It started immediately after I updated my phone."*
Without the prior conversation, this could be about any issue. The classifier defaults to `software_update_issue` based on "update", but the actual problem might be battery, apps, or connectivity.

**Hypothesis**: Multi-turn context is needed. Our single-turn pipeline will misclassify or under-specify follow-ups.

### 4. Template drafts feel generic for specific problems
**Example**: Customer reports specific Bluetooth issue with AirPods Pro on iOS 15.2. Template reply says "please DM us your device info" — but the customer already provided it.

**Hypothesis**: Template drafts ignore the customer's message content. LLM RAG drafting can acknowledge specific details and skip redundant requests.

### 5. Over-escalation on normal device mentions
**Example**: *"My iPhone 7 is really slow since the update"*
The heuristic classifier assigns this high confidence, but the `has_order_reference` check sees digits in "7" and passes, while "iPhone 12" or "MacBook Pro" without version numbers may trigger unnecessary escalation for "missing reference."

**Hypothesis**: The `has_order_reference` regex is tuned for order/tracking numbers, not device model numbers. Needs a device-model whitelist.

## 6. What Is Misleading About My Headline Number?

**The 100% accuracy and 1.0 macro-F1 are circular and meaningless.**

The golden set was auto-labeled by the keyword heuristic classifier. When the same classifier is evaluated against its own labels, it perfectly agrees with itself. This is **not** a measure of classification quality — it's a measure of self-consistency.

**To fix this**: The golden set labels must be manually reviewed and corrected by a human. After human correction, the heuristic classifier will show its true error rate, and the LLM classifier should outperform it.

**Other misleading aspects**:
- **Stratified sampling inflates rare-class performance**: The golden set has 18 `account_or_security` examples (9%) vs. 3.6% in the natural data. Per-intent F1 for rare classes looks better than production reality.
- **Auto-handle rate ≠ safety**: A high auto-handle rate sounds efficient, but each false auto-handle risks a bad customer experience (weighted 10× in our cost metric).
- **Rule-based judge cannot assess semantic quality**: It checks for numeric hallucination and keyword presence, not whether the reply actually helps the customer.
- **Small golden set → wide confidence intervals**: With 197 examples, a 95% CI on accuracy spans roughly ±7 percentage points.

## 7. What I'd Do Next With One More Week

1. **Manually correct all 197 golden labels** — break the circular evaluation (~3 hours)
2. **Connect Gemini API** — enable LLM classification and RAG drafting (via gemini-2.0-flash) for a true full-system evaluation
3. **Embedding-based retrieval** — replace token overlap with sentence embeddings (e.g. text-embedding-004) for better semantic matching
4. **LLM judge calibration** — run the LLM judge on 50 examples, independently score the same 50 as a human, compute Cohen's kappa
5. **Multi-turn context window** — concatenate the previous 2–3 turns as context for classification and drafting
6. **A/B test framework** — serve both template and LLM drafts, track which humans choose to send
7. **Expand to a second brand** — validate the pipeline generalizes beyond AppleSupport
