"""Shared policy configuration for the AppleSupport copilot.

The intent taxonomy is derived from keyword-frequency analysis of ~5K
AppleSupport conversations in the TWCS dataset. It covers six mutually
exclusive categories that together account for >95% of inbound messages.
"""

# ── Intent taxonomy ──────────────────────────────────────────────────
INTENTS = (
    "software_update_issue",     # iOS update broke something, post-update regressions
    "battery_or_power",          # battery drain, charging, power-off
    "app_or_performance",        # app crashes, freezes, lag, slow device
    "connectivity",              # wifi, bluetooth, cellular, signal problems
    "account_or_security",       # Apple ID, password, locked out, 2FA
    "general_inquiry",           # how-to, product questions, feedback, other
)

# ── Escalation policy ────────────────────────────────────────────────
# These intents always require human review (identity / financial risk)
SENSITIVE = {"account_or_security"}

# These intents need a concrete reference to be auto-handled
ACTIONABLE = {"battery_or_power", "app_or_performance", "connectivity"}

# Confidence below this threshold triggers escalation
CONF_THRESHOLD = 0.72
