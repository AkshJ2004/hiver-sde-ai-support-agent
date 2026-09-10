# Goal

Build a conservative AppleSupport triage + draft-assist copilot using the
Customer Support on Twitter dataset (2.8M tweets, ~107K AppleSupport
conversations).

The system classifies customer messages into six intents, retrieves
historically similar brand replies for grounding, drafts context-aware
responses, and routes sensitive or uncertain cases for human review.

It is evaluated against a 197-example golden set with three system
configurations (trivial baseline, keyword baseline, LLM-powered) and scored
on classification accuracy, decision safety (weighted false-auto-handle cost),
and reply quality (5-dimension judge rubric).

This is a *copilot*, not an autonomous agent — no message is sent without
human approval.
