# app/api/routes/recent_applications.py
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.dependencies import role_required
from app.services.recent_applications_service import list_recent_applications

router = APIRouter(prefix="/applications", tags=["Applications"])


class RecentApplicationOut(BaseModel):
    user_cv_id: int
    user_id: int
    name: str
    email: EmailStr
    job_type: Optional[str] = None
    job_position: Optional[str] = None
    cv_url: str
    created_at: Optional[datetime] = None


@router.get("/recent", response_model=List[RecentApplicationOut])
def get_recent_applications(
    _: None = Depends(role_required("admin")),  # admin-only
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    since_days: Optional[int] = Query(
        None, ge=1, description="Only include applications from the last N days (requires created_at on UserCV)"
    ),
) -> List[RecentApplicationOut]:
    """
    Returns recent applications (UserCV rows), newest first.
    - If your schema has UserCV.created_at, we sort/filter by it.
    - Otherwise we sort by UserCV.id (as a recency proxy).
    """
    data = list_recent_applications(db, limit=limit, offset=offset, since_days=since_days)
    return [RecentApplicationOut(**item) for item in data]
