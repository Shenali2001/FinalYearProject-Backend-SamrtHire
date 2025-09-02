from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from app.database import Base, engine
import app.models  # <-- IMPORTANT: registers InterviewReport, UserCV, etc.

# Create tables after models are imported
Base.metadata.create_all(bind=engine)

from app.api import auth, jobs, cv, interview  # routers AFTER create_all

from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-hire")

app = FastAPI(title="AI Interview System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    logger.info(f"[startup] device={device}")

    BASE = Path(__file__).parent
    qg_dir  = BASE / "question_generation_model"

    # --- T5 (QG) ---
    if qg_dir.is_dir():
        t5_tokenizer = T5Tokenizer.from_pretrained(str(qg_dir))
        t5_model = T5ForConditionalGeneration.from_pretrained(str(qg_dir)).to(device)
        logger.info(f"[startup] T5 loaded from {qg_dir}")
    else:
        logger.warning("[startup] T5 finetuned folder not found under app/. Falling back to t5-small.")
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)



    app.state.device = device
    app.state.t5_tokenizer = t5_tokenizer
    app.state.t5_model = t5_model


@app.get("/", tags=["Health"])
async def root():
    return {"message": "AI Interview System API is running!"}

# Routers
app.include_router(auth.router)      # /auth
app.include_router(jobs.router)      # /jobs
app.include_router(cv.router)        # /cv
app.include_router(interview.router) # /interview
