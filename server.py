"""FastAPI backend server for the AppleSupport Copilot Web Application."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv()

from pipeline.constants import CONF_THRESHOLD, INTENTS
from pipeline.core import infer, retrieve
from pipeline.judge import judge as run_judge

ROOT = Path(__file__).resolve().parent
PAIR_PATH = ROOT / "data" / "processed" / "apple_pairs.jsonl"
WEB_DIR = ROOT / "web"

app = FastAPI(title="AppleSupport Copilot API", version="1.0.0")

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Knowledge Base into memory once
PAIRS: list[dict] = []
if PAIR_PATH.exists():
    with open(PAIR_PATH, encoding="utf-8") as f:
        PAIRS = [json.loads(line) for line in f if line.strip()]


class InferRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer inquiry message")
    threshold: float | str = Field("auto", description="'auto' for situational adaptive threshold or manual float 0.50-0.99")
    provider: str = Field("gemini", description="'gemini', 'heuristic', or 'majority'")


@app.get("/api/status")
def get_status():
    """System health, model configurations, and KB status."""
    return {
        "status": "online",
        "provider": os.getenv("PROVIDER", "gemini"),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "kb_pairs_count": len(PAIRS),
        "intents": list(INTENTS),
        "default_threshold": "auto",
    }


@app.get("/api/samples")
def get_samples(limit: int = 50):
    """Retrieve real customer conversations from the TWCS dataset."""
    samples = []
    for item in PAIRS[:limit]:
        samples.append({
            "id": item.get("id"),
            "customer": item.get("customer"),
            "intent": item.get("intent", ""),
            "actual_reply": item.get("brand_reply", ""),
        })
    return {"samples": samples, "total": len(PAIRS)}


@app.post("/api/infer")
def run_inference(req: InferRequest):
    """Run full pipeline: classify -> retrieve -> draft -> decide -> judge."""
    customer_text = req.text.strip()
    if not customer_text:
        raise HTTPException(status_code=400, detail="Customer message cannot be empty.")

    provider_to_use = req.provider
    if provider_to_use == "gemini" and not os.getenv("GEMINI_API_KEY"):
        # Fallback to heuristic if API key is not present
        provider_to_use = "heuristic"

    try:
        result = infer(
            customer=customer_text,
            kb=PAIRS,
            brand="AppleSupport",
            threshold=req.threshold,
            provider_override=provider_to_use,
        )
    except Exception as e:
        # Graceful fallback to heuristic if Gemini network or quota issue occurs
        try:
            result = infer(
                customer=customer_text,
                kb=PAIRS,
                brand="AppleSupport",
                threshold=req.threshold,
                provider_override="heuristic",
            )
            result["warning"] = f"LLM error ('{str(e)}'); fell back to heuristic baseline."
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=str(inner_e))

    # Retrieve grounding evidence
    evidence = retrieve(customer_text, result["intent"], PAIRS, brand="AppleSupport", k=3)

    # Run judge evaluation
    judge_res = run_judge(customer_text, result["intent"], result["draft_reply"], evidence, provider=provider_to_use)

    return {
        "infer": result,
        "judge": judge_res,
        "evidence": evidence,
    }


# Serve static web frontend
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page HTML interface."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Web UI is building..."}
