# app/services/stats_service.py
from typing import Dict
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.enums import UserRole
from app.models.user import User
from app.models.cv import UserCV
from app.models.job import JobPosition


def get_overview_counts(db: Session) -> Dict[str, int]:
    """
    Returns three metrics:
      - candidate_users: number of users with role == CANDIDATE
      - applications: total number of applications (counts rows in UserCV)
      - job_positions: total number of job positions (rows in JobPosition)
    """
    candidate_users = db.query(func.count(User.id))\
                        .filter(User.role == UserRole.CANDIDATE)\
                        .scalar() or 0

    applications = db.query(func.count(UserCV.id)).scalar() or 0

    job_positions = db.query(func.count(JobPosition.id)).scalar() or 0

    return {
        "candidate_users": int(candidate_users),
        "applications": int(applications),
        "job_positions": int(job_positions),
    }
