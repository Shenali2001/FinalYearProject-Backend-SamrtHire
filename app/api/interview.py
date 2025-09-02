# app/api/interview_reports.py
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.services.cv_service import (
    start_adaptive_interview_service,
    submit_adaptive_answer_service,
)
from app.database import get_db  # <-- adjust if your get_db lives elsewhere

router = APIRouter(prefix="/interview", tags=["interview"])

@router.post("/start")
def start_interview(
    request: Request,
    email: str = Query(..., description="Candidate email used to look up CV"),
    max_questions: int = Query(10, ge=1, le=30, description="Max technical Qs after icebreaker"),
    db: Session = Depends(get_db),
):
    """
    Starts an adaptive interview. Returns the icebreaker immediately.
    """
    return start_adaptive_interview_service(
        email=email,
        db=db,
        app=request.app,
        max_questions=max_questions,
    )

@router.post("/answer")
def submit_answer(
    request: Request,
    email: str = Query(..., description="Candidate email (interview session key)"),
    answer: str = Query(..., description="Candidate's answer to the previous question"),
    question: str | None = Query(None, description="Echo of the question just answered (optional)"),
    db: Session = Depends(get_db),
):
    """
    Submits an answer and returns either the next question or the final report.
    """
    return submit_adaptive_answer_service(
        email=email,
        question=question,
        answer=answer,
        app=request.app,
        db=db,                     # <-- the missing argument
    )
