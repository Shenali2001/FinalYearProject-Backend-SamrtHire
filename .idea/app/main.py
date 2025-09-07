from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from app.database import Base, engine
import app.models  # registers InterviewReport, UserCV, etc.

# Create tables after models are imported
Base.metadata.create_all(bind=engine)

from app.api import (
    auth, jobs, cv, interview, applications, users, stats,
    recent_applications, admin_candidates
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-hire")

app = FastAPI(title="AI Interview System", version="1.0.0")

# CORS (adjust origins via env if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ALLOW_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@app.on_event("startup")
async def startup_event():
    device = _pick_device()
    logger.info(f"[startup] device={device}")

    # Where your fine-tuned checkpoint lives (can override with env)
    # Default matches your training save folder
    ckpt_env = os.getenv("QG_MODEL_PATH", "").strip()
    if ckpt_env:
        ckpt_path = Path(ckpt_env).expanduser()
    else:
        # fallback to a sibling folder next to this file
        base_dir = Path(__file__).parent
        ckpt_path = (base_dir / "technical_qg_enhanced_5986").resolve()

    # Load T5 (QG)
    try:
        if ckpt_path.is_dir():
            t5_tokenizer = T5Tokenizer.from_pretrained(str(ckpt_path))
            t5_model = T5ForConditionalGeneration.from_pretrained(str(ckpt_path)).to(device)
            logger.info(f"[startup] T5 loaded from {ckpt_path}")
        else:
            # If your finetuned folder isn't found, fall back to base FLAN-T5
            logger.warning(f"[startup] Finetuned folder not found at {ckpt_path}. Falling back to 'google/flan-t5-base'.")
            t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
    except Exception as e:
        # Last-resort fallback
        logger.exception(f"[startup] Failed to load QG model from '{ckpt_path}': {e}")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        logger.warning("[startup] Loaded fallback 't5-small'.")

    t5_model.eval()

    # Expose to services (cv_service uses app.state.* if present)
    app.state.device = device
    app.state.t5_tokenizer = t5_tokenizer
    app.state.t5_model = t5_model
    app.state.qg_model_path = str(ckpt_path) if ckpt_path else "google/flan-t5-base"

@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI Interview System API is running!"}

# Routers
app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(cv.router)
app.include_router(interview.router)
app.include_router(applications.router)
app.include_router(users.router)
app.include_router(stats.router)
app.include_router(recent_applications.router)
app.include_router(admin_candidates.router)
