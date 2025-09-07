# app/services/candidate_service.py
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, asc, desc

from app.models.enums import UserRole
from app.models.user import User
from app.models.cv import UserCV


ALLOWED_SORT_COLUMNS = {
    "created_at": User.created_at,
    "name": User.name,
    "email": User.email,
}


def list_candidates(
    db: Session,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
) -> Dict[str, object]:
    """
    Returns paginated candidate users with an applications count:
      - items: [{id, name, email, phone_number, created_at, applications}]
      - total: total matching rows (for pagination)
    """
    # Base filter: only candidate users
    base = db.query(User).filter(User.role == UserRole.CANDIDATE)

    # Optional search on name/email
    if q:
        like = f"%{q}%"
        base = base.filter((User.name.ilike(like)) | (User.email.ilike(like)))

    total = base.count()

    # Sorting
    sort_col = ALLOWED_SORT_COLUMNS.get(sort_by, User.created_at)
    order_expr = desc(sort_col) if str(sort_dir).lower() == "desc" else asc(sort_col)

    # Query items with applications count
    rows = (
        base
        .outerjoin(UserCV, UserCV.user_id == User.id)
        .with_entities(
            User.id.label("id"),
            User.name.label("name"),
            User.email.label("email"),
            User.phone_number.label("phone_number"),
            User.created_at.label("created_at"),
            func.count(UserCV.id).label("applications"),
        )
        .group_by(User.id, User.name, User.email, User.phone_number, User.created_at)
        .order_by(order_expr)
        .limit(page_size)
        .offset((page - 1) * page_size)
        .all()
    )

    items: List[Dict[str, object]] = [
        {
            "id": r.id,
            "name": r.name,
            "email": r.email,
            "phone_number": r.phone_number,
            "created_at": r.created_at,
            "applications": int(r.applications or 0),
        }
        for r in rows
    ]

    return {"items": items, "total": int(total)}
