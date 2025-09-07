# app/services/application_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.models.enums import UserRole  # if you need roles elsewhere
from app.models.user import User
from app.models.interview_reports import InterviewReport
from app.models.cv import UserCV
from app.models.job import JobType
from app.models.job import JobPosition


def list_applications(db: Session) -> List[Dict[str, Optional[Any]]]:
    """
    Returns one entry per application (per UserCV) with:
      - user_id, user_cv_id
      - name, email
      - job_type (name), job_position (name)
      - cv_url
      - feedback (latest InterviewReport.summary for that CV)
    Only users who have at least one UserCV are included.
    """
    # Subquery: get the most recent InterviewReport per user_cv_id
    latest_report_sq = (
        db.query(
            InterviewReport.user_cv_id.label("user_cv_id"),
            func.max(InterviewReport.created_at).label("max_created_at"),
        )
        .group_by(InterviewReport.user_cv_id)
        .subquery()
    )

    IR = aliased(InterviewReport)

    rows = (
        db.query(
            User.id.label("user_id"),
            UserCV.id.label("user_cv_id"),
            User.name.label("name"),
            User.email.label("email"),
            JobType.name.label("job_type"),
            JobPosition.name.label("job_position"),
            UserCV.cv_url.label("cv_url"),
            IR.summary.label("feedback"),
        )
        .join(UserCV, UserCV.user_id == User.id)                    # users who applied (have a CV)
        .outerjoin(JobType, JobType.id == UserCV.job_type_id)
        .outerjoin(JobPosition, JobPosition.id == UserCV.job_position_id)
        .outerjoin(latest_report_sq, latest_report_sq.c.user_cv_id == UserCV.id)
        .outerjoin(
            IR,
            (IR.user_cv_id == latest_report_sq.c.user_cv_id)
            & (IR.created_at == latest_report_sq.c.max_created_at),
        )
        .order_by(UserCV.id.desc())
        .all()
    )

    # Convert result rows into plain dicts for the API layer
    return [
        {
            "user_id": r.user_id,
            "user_cv_id": r.user_cv_id,
            "name": r.name,
            "email": r.email,
            "job_type": r.job_type,
            "job_position": r.job_position,
            "cv_url": r.cv_url,
            "feedback": r.feedback,
        }
        for r in rows
    ]

def delete_application_by_cv_id(db: Session, user_cv_id: int) -> Optional[Dict]:
    """
    Deletes a single UserCV by ID.
    - Cascades to Education/Skill/Project/WorkExperience (already configured on UserCV).
    - Also deletes InterviewReports linked to this UserCV.
    Returns a summary dict or None if not found.
    """
    cv = (
        db.query(UserCV)
        .options(
            selectinload(UserCV.education),
            selectinload(UserCV.skills),
            selectinload(UserCV.projects),
            selectinload(UserCV.work_experience),
            selectinload(UserCV.user),
        )
        .filter(UserCV.id == user_cv_id)
        .first()
    )

    if not cv:
        return None

    counts = {
        "education": len(cv.education or []),
        "skills": len(cv.skills or []),
        "projects": len(cv.projects or []),
        "work_experience": len(cv.work_experience or []),
    }

    deleted_reports = (
        db.query(InterviewReport)
        .filter(InterviewReport.user_cv_id == user_cv_id)
        .delete(synchronize_session=False)
    )

    payload = {
        "user_cv_id": cv.id,
        "user_id": cv.user_id,
        "deleted_interview_reports": int(deleted_reports or 0),
        "counts": counts,
    }

    db.delete(cv)   # cascades handle child tables
    db.commit()

    return payload
