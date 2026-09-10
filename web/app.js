/**
 * AppleSupport Copilot Studio - Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const kbCount = document.getElementById("kb-count");

  const messageInput = document.getElementById("message-input");
  const charCount = document.getElementById("char-count");
  const clearBtn = document.getElementById("clear-btn");

  const labelGemini = document.getElementById("label-gemini");
  const labelHeuristic = document.getElementById("label-heuristic");

  // Threshold controls
  const manualOverrideToggle = document.getElementById("manual-override-toggle");
  const manualSliderContainer = document.getElementById("manual-slider-container");
  const adaptiveModeBadge = document.getElementById("adaptive-mode-badge");
  const thresholdCaption = document.getElementById("threshold-caption");
  const thresholdSlider = document.getElementById("threshold-slider");
  const thresholdVal = document.getElementById("threshold-val");

  const analyzeBtn = document.getElementById("analyze-btn");

  const emptyState = document.getElementById("empty-state");
  const resultsContent = document.getElementById("results-content");
  const providerBadge = document.getElementById("provider-badge");

  // Results Elements
  const decisionCard = document.getElementById("decision-card");
  const decisionPill = document.getElementById("decision-pill");
  const decisionTitle = document.getElementById("decision-title");
  const decisionReason = document.getElementById("decision-reason");

  const intentTitle = document.getElementById("intent-title");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceVal = document.getElementById("confidence-val");
  const metaThreshold = document.getElementById("meta-threshold");
  const situationalRiskText = document.getElementById("situational-risk-text");

  const draftReplyText = document.getElementById("draft-reply-text");
  const copyDraftBtn = document.getElementById("copy-draft-btn");

  const judgeTypePill = document.getElementById("judge-type-pill");
  const judgeRationale = document.getElementById("judge-rationale");
  const sendableBadge = document.getElementById("sendable-badge");

  const evidenceList = document.getElementById("evidence-list");
  const jsonDebug = document.getElementById("json-debug");

  // 1. Load System Status
  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) throw new Error("Backend offline");
      const data = await res.json();

      statusPill.className = "status-pill online";
      if (data.has_gemini_key) {
        statusText.textContent = `${data.model || "Gemini 3.6 Flash"} Online`;
      } else {
        statusText.textContent = "Heuristic Baseline Active";
      }
      if (data.kb_pairs_count) {
        kbCount.textContent = data.kb_pairs_count.toLocaleString();
      }
    } catch (err) {
      statusPill.className = "status-pill checking";
      statusText.textContent = "Server Offline";
    }
  }

  // 2. Character counter
  function updateCharCount() {
    const len = messageInput.value.length;
    charCount.textContent = `${len} character${len === 1 ? "" : "s"}`;
  }
  messageInput.addEventListener("input", updateCharCount);

  // Clear button
  clearBtn.addEventListener("click", () => {
    messageInput.value = "";
    updateCharCount();
    messageInput.focus();
  });

  // Quick Test Scenario Chips
  const scenarioChips = document.querySelectorAll(".scenario-chip");
  scenarioChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const text = chip.getAttribute("data-text") || "";
      messageInput.value = text;
      updateCharCount();
      messageInput.focus();
    });
  });

  // Threshold Override Toggle
  manualOverrideToggle.addEventListener("change", (e) => {
    const isManual = e.target.checked;
    manualSliderContainer.classList.toggle("hidden", !isManual);
    if (isManual) {
      adaptiveModeBadge.textContent = "MANUAL OVERRIDE";
      adaptiveModeBadge.classList.add("manual");
      thresholdCaption.textContent = `Using static threshold (${thresholdSlider.value}%). Disables situational risk adaptation.`;
    } else {
      adaptiveModeBadge.textContent = "AUTO-CALIBRATED";
      adaptiveModeBadge.classList.remove("manual");
      thresholdCaption.textContent = "Automatically sets confidence safety margins based on inquiry risk (55% routine accessories, 68% tech issues, 85% sensitive security).";
    }
  });

  // Slider change
  thresholdSlider.addEventListener("input", (e) => {
    thresholdVal.textContent = `${e.target.value}%`;
    if (manualOverrideToggle.checked) {
      thresholdCaption.textContent = `Using static threshold (${e.target.value}%). Disables situational risk adaptation.`;
    }
  });

  // Radio toggle styling
  const radioInputs = document.querySelectorAll('input[name="provider"]');
  radioInputs.forEach(input => {
    input.addEventListener("change", () => {
      labelGemini.classList.toggle("active", input.value === "gemini" && input.checked);
      labelHeuristic.classList.toggle("active", input.value === "heuristic" && input.checked);
    });
  });

  // Copy Draft button
  copyDraftBtn.addEventListener("click", () => {
    const text = draftReplyText.textContent.replace(/^"|"$/g, "").trim();
    navigator.clipboard.writeText(text).then(() => {
      const orig = copyDraftBtn.textContent;
      copyDraftBtn.textContent = "✅ Copied!";
      setTimeout(() => { copyDraftBtn.textContent = orig; }, 1800);
    });
  });

  // 3. Run Inference
  async function runInference() {
    const text = messageInput.value.trim();
    if (!text) {
      alert("Please enter a customer inquiry message first (or click a quick test scenario).");
      messageInput.focus();
      return;
    }

    const selectedProvider = document.querySelector('input[name="provider"]:checked')?.value || "gemini";
    const isManual = manualOverrideToggle.checked;
    const thresholdParam = isManual ? (parseFloat(thresholdSlider.value) / 100.0) : "auto";

    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;

    try {
      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          provider: selectedProvider,
          threshold: thresholdParam,
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      renderResults(data, selectedProvider);
    } catch (err) {
      alert(`Inference failed: ${err.message}`);
    } finally {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.disabled = false;
    }
  }

  analyzeBtn.addEventListener("click", runInference);

  // Keyboard shortcut: Ctrl+Enter or Cmd+Enter
  messageInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      runInference();
    }
  });

  // 4. Render Results View
  function renderResults(data, provider) {
    const infer = data.infer || {};
    const judge = data.judge || {};
    const evidence = data.evidence || [];

    // Switch view from empty to results
    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Provider Badge
    providerBadge.textContent = `Provider: ${infer.provider || provider}`;

    // Card 1: Decision
    const isAuto = infer.decision === "auto_handle";
    decisionPill.className = `decision-pill ${isAuto ? "safe" : "risk"}`;
    decisionPill.textContent = isAuto ? "ELIGIBLE FOR AUTO-HANDLING" : "HUMAN ESCALATION REQUIRED";
    decisionTitle.textContent = isAuto ? "Auto-Handle Eligible" : "Escalate to Senior Human";
    decisionReason.textContent = infer.decision_reason || "Deterministic safety gate policy";

    // Card 2: Intent & Confidence & Situational Threshold
    const cleanIntent = (infer.intent || "general_inquiry").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    intentTitle.textContent = cleanIntent;

    const confPct = Math.round((infer.confidence || 0.5) * 100);
    confidenceBar.style.width = `${Math.min(confPct, 100)}%`;
    confidenceVal.textContent = `${confPct}%`;

    const effThresh = infer.effective_threshold !== undefined ? infer.effective_threshold : 0.68;
    const threshPct = Math.round(effThresh * 100);
    const isAutoMode = infer.threshold_mode === "auto";
    metaThreshold.textContent = isAutoMode ? `${threshPct}% (Auto)` : `${threshPct}% (Manual)`;

    if (situationalRiskText) {
      situationalRiskText.textContent = infer.risk_tier || "Standard Technical Support";
    }

    // Card 3: Draft Reply
    draftReplyText.textContent = `"${infer.draft_reply || "No draft generated."}"`;

    // Reply Quality Rubric
    judgeTypePill.textContent = judge.judge_type === "llm" ? "Gemini 3.6 Flash Judge" : "Rule-Based Judge";
    const dims = ["groundedness", "brand_voice", "actionability", "empathy", "safety"];
    dims.forEach(dim => {
      const score = judge[dim] ?? 3;
      const scoreEl = document.getElementById(`score-${dim.replace(/_/g, "-")}`);
      const meterEl = document.getElementById(`meter-${dim.replace(/_/g, "-")}`);
      if (scoreEl) scoreEl.textContent = `${score}/5`;
      if (meterEl) meterEl.style.width = `${(score / 5) * 100}%`;
    });

    const isSendable = !!judge.overall_sendable;
    sendableBadge.className = `sendable-badge ${isSendable ? "pass" : "fail"}`;
    sendableBadge.textContent = isSendable ? "YES" : "NO";
    judgeRationale.textContent = judge.rationale ? `Judge rationale: ${judge.rationale}` : "";

    // Historical Grounding Evidence
    evidenceList.innerHTML = "";
    if (evidence && evidence.length > 0) {
      evidence.forEach((ev, i) => {
        const card = document.createElement("div");
        card.className = "evidence-card";
        const evIntent = (ev.intent || infer.intent || "").replace(/_/g, " ");
        card.innerHTML = `
          <div class="evidence-meta">
            <span>Case Reference #${ev.id || i + 1}</span>
            <span>Benchmark Match: ${escapeHtml(evIntent)}</span>
          </div>
          <div class="dialogue-turn">
            <span class="speaker">Customer Inquiry:</span>
            <p>${escapeHtml(ev.customer)}</p>
          </div>
          <div class="dialogue-turn brand">
            <span class="speaker">Verified Historical Resolution:</span>
            <p>${escapeHtml(ev.brand_reply || ev.reply || "")}</p>
          </div>
        `;
        evidenceList.appendChild(card);
      });
    } else {
      evidenceList.innerHTML = '<p class="section-sub">No similar historical resolutions found in knowledge base.</p>';
    }

    // JSON Debugger
    jsonDebug.textContent = JSON.stringify(data, null, 2);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Set default initial custom inquiry
  messageInput.value = "My iPhone case is broken and cracked, where can I buy a replacement in the Apple Store?";
  updateCharCount();

  // Initialize status
  loadStatus();
});
