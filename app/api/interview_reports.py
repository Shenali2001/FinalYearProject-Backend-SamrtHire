from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.database import get_db
from app.models.interview_reports import InterviewReport, InterviewReportRead

router = APIRouter(prefix="/interview-reports", tags=["Interview Reports"])

@router.get("/by-cv/{user_cv_id}", response_model=List[InterviewReportRead])
def list_reports_by_user_cv_id(
    user_cv_id: int,
    status: Optional[str] = Query(None, description="Filter by status (e.g., finished, pending)"),
    limit: int = Query(50, ge=1, le=200, description="Max number of results"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    db: Session = Depends(get_db),
):
    """
    Return all interview reports for the given user_cv_id.
    Supports optional status filtering and pagination.
    Results are ordered by created_at (newest first).
    """
    stmt = (
        select(InterviewReport)
        .where(InterviewReport.user_cv_id == user_cv_id)
        .order_by(desc(InterviewReport.created_at))
        .offset(offset)
        .limit(limit)
    )

    if status is not None:
        stmt = stmt.where(InterviewReport.status == status)

    results = db.execute(stmt).scalars().all()

    # Optional: if you want a 404 when none found for that CV id
    # if not results:
    #     raise HTTPException(status_code=404, detail="No reports found for this user_cv_id")

    return results
