# app/api/routes/stats.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.stats_service import get_overview_counts
from app.auth.dependencies import role_required


router = APIRouter(prefix="/stats", tags=["Stats"])

class StatsOverviewOut(BaseModel):
    candidate_users: int
    applications: int
    job_positions: int

@router.get("/overview", response_model=StatsOverviewOut)
def stats_overview(
    _: None = Depends(role_required("admin")),
    db: Session = Depends(get_db),
) -> StatsOverviewOut:
    """
    Returns:
      - candidate_users: users with role == CANDIDATE
      - applications: total number of applications (UserCV rows)
      - job_positions: total number of job positions
    """
    data = get_overview_counts(db)
    return StatsOverviewOut(**data)
