# app/services/recent_applications_service.py
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.models.user import User
from app.models.cv import UserCV
from app.models.job import JobType
from app.models.job import JobPosition


def list_recent_applications(
    db: Session,
    limit: int = 20,
    offset: int = 0,
    since_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns recent applications (UserCV rows) ordered by most recent.
    - If UserCV.created_at exists, order/filter by it.
    - Else, fall back to UserCV.id as a recency proxy.
    """
    has_created_at = "created_at" in UserCV.__table__.c  # works even if attribute not on class

    q = (
        db.query(
            UserCV.id.label("user_cv_id"),
            User.id.label("user_id"),
            User.name.label("name"),
            User.email.label("email"),
            JobType.name.label("job_type"),
            JobPosition.name.label("job_position"),
            UserCV.cv_url.label("cv_url"),
            (UserCV.__table__.c.get("created_at")).label("created_at") if has_created_at else None,
        )
        .join(User, User.id == UserCV.user_id)
        .outerjoin(JobType, JobType.id == UserCV.job_type_id)
        .outerjoin(JobPosition, JobPosition.id == UserCV.job_position_id)
    )

    if has_created_at and since_days:
        # Filter by created_at >= now - since_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        q = q.filter(UserCV.__table__.c.created_at >= cutoff)

    # Order by most recent
    order_col = UserCV.__table__.c.created_at if has_created_at else UserCV.id
    q = q.order_by(desc(order_col)).limit(limit).offset(offset)

    rows = q.all()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "user_cv_id": r.user_cv_id,
                "user_id": r.user_id,
                "name": r.name,
                "email": r.email,
                "job_type": r.job_type,
                "job_position": r.job_position,
                "cv_url": r.cv_url,
                "created_at": getattr(r, "created_at", None),
            }
        )
    return results
