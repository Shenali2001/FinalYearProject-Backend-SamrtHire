# app/services/user_service.py
from typing import Dict, List, Optional
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_

from app.models.user import User
from app.models.cv import UserCV
from app.models.interview_reports import InterviewReport


class DeleteUserResult(Dict):
    """
    Simple dict-based result so the API layer can wrap it in Pydantic if desired.
    Keys:
      - user_id: int
      - deleted_user: bool
      - deleted_user_cv_ids: List[int]
      - deleted_interview_reports: int
      - counts: {education, skills, projects, work_experience}
    """
    pass


def delete_user_and_details(db: Session, user_id: int) -> DeleteUserResult:
    """
    Delete the user and all related data:
      - Interview reports linked to the user or any of the user's CVs
      - Each UserCV (and via ORM cascade on UserCV: education, skills, projects, work_experience)
      - The User itself

    Notes:
      - We DO NOT use bulk delete for UserCVs because we want relationship cascades
        (Education/Skill/Project/WorkExperience) to fire properly.
      - InterviewReport FKs are defined with ondelete="SET NULL"; since you want them
        *removed*, we explicitly delete them here.
    """
    # Load the user and pre-load CVs & their nested relations for counting + proper deletes
    user: Optional[User] = (
        db.query(User)
        .options(
            selectinload(User.cvs)
            .selectinload(UserCV.education),
            selectinload(User.cvs)
            .selectinload(UserCV.skills),
            selectinload(User.cvs)
            .selectinload(UserCV.projects),
            selectinload(User.cvs)
            .selectinload(UserCV.work_experience),
        )
        .filter(User.id == user_id)
        .first()
    )

    if not user:
        # Let the API layer turn this into a 404; returning a sentinel is fine too.
        return DeleteUserResult(
            user_id=user_id,
            deleted_user=False,
            deleted_user_cv_ids=[],
            deleted_interview_reports=0,
            counts={"education": 0, "skills": 0, "projects": 0, "work_experience": 0},
        )

    # Snapshot related IDs and counts BEFORE deletion
    cv_ids: List[int] = [cv.id for cv in user.cvs]
    counts = {
        "education": sum(len(cv.education or []) for cv in user.cvs),
        "skills": sum(len(cv.skills or []) for cv in user.cvs),
        "projects": sum(len(cv.projects or []) for cv in user.cvs),
        "work_experience": sum(len(cv.work_experience or []) for cv in user.cvs),
    }

    # Start a transaction
    # (FastAPI's default session pattern usually commits/rolls back automatically,
    # but being explicit makes intent clear.)
    # Delete interview reports tied to the user or any of their CVs
    deleted_reports = (
        db.query(InterviewReport)
        .filter(
            or_(
                InterviewReport.user_id == user_id,
                (InterviewReport.user_cv_id.in_(cv_ids) if cv_ids else False),
            )
        )
        .delete(synchronize_session=False)
    )

    # Delete each CV (this will cascade to Education/Skills/Projects/WorkExperience)
    for cv in user.cvs:
        db.delete(cv)

    # Finally delete the user
    db.delete(user)

    # Commit the transaction
    db.commit()

    return DeleteUserResult(
        user_id=user_id,
        deleted_user=True,
        deleted_user_cv_ids=cv_ids,
        deleted_interview_reports=deleted_reports or 0,
        counts=counts,
    )
