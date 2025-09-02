# app/api/routes/admin_candidates.py
import math
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.candidate_service import list_candidates
from app.auth.dependencies import role_required


router = APIRouter(prefix="/admin/candidates", tags=["Admin - Candidates"])


class CandidateOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    phone_number: Optional[str] = None
    created_at: datetime
    applications: int


class CandidateListOut(BaseModel):
    items: List[CandidateOut]
    total: int
    page: int
    page_size: int
    pages: int


@router.get("", response_model=CandidateListOut)
def get_candidates_for_admin(
    _: None = Depends(role_required("admin")),  # admin-only
    db: Session = Depends(get_db),
    q: Optional[str] = Query(None, description="Search by name or email"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", pattern="^(created_at|name|email)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
) -> CandidateListOut:
    """
    List candidate users (role == CANDIDATE) with:
      - search: q (name/email)
      - sort: sort_by in {created_at, name, email}, sort_dir {asc, desc}
      - pagination: page, page_size
    Includes an `applications` count (number of UserCVs per user).
    """
    data = list_candidates(
        db=db,
        q=q,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
    pages = max(1, math.ceil(data["total"] / page_size)) if data["total"] else 1
    return CandidateListOut(
        items=[CandidateOut(**item) for item in data["items"]],
        total=data["total"],
        page=page,
        page_size=page_size,
        pages=pages,
    )
