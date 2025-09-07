# app/api/routes/applications.py
from typing import List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Dict
from fastapi import HTTPException, status

from app.database import get_db
from app.services.application_service import list_applications as list_applications_svc
from app.services.application_service import delete_application_by_cv_id

router = APIRouter(prefix="/applications", tags=["Applications"])


class ApplicationOut(BaseModel):
    user_id: int
    user_cv_id: int
    name: str
    email: EmailStr
    job_type: Optional[str] = None
    job_position: Optional[str] = None
    cv_url: str
    feedback: Optional[str] = None

    class Config:
        # For Pydantic v2; if you're on v1, use: orm_mode = True
        from_attributes = True


@router.get("", response_model=List[ApplicationOut])
def list_applications(db: Session = Depends(get_db)) -> List[ApplicationOut]:
    """
    Get all applications (one per UserCV) including user info, job type/position,
    CV URL, and latest feedback (InterviewReport.summary).
    """
    data = list_applications_svc(db)
    return [ApplicationOut(**item) for item in data]


class DeleteApplicationOut(BaseModel):
    user_cv_id: int
    user_id: int
    deleted_interview_reports: int
    counts: Dict[str, int]

@router.delete("/{user_cv_id}", response_model=DeleteApplicationOut)
def delete_application(user_cv_id: int, db: Session = Depends(get_db)) -> DeleteApplicationOut:
    """
    Delete a specific application (UserCV) by ID.
    - Also deletes InterviewReports linked to this UserCV.
    - Child tables (education/skills/projects/work_experience) are removed via cascade.
    """
    result = delete_application_by_cv_id(db, user_cv_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"UserCV with id {user_cv_id} not found.",
        )
    return DeleteApplicationOut(**result)