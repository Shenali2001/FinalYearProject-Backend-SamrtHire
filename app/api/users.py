# app/api/routes/users.py
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.user_service import delete_user_and_details

router = APIRouter(prefix="/users", tags=["Users"])


class DeleteUserResponse(BaseModel):
    user_id: int
    deleted_user: bool
    deleted_user_cv_ids: List[int]
    deleted_interview_reports: int
    counts: Dict[str, int]


@router.delete("/{user_id}", response_model=DeleteUserResponse, status_code=status.HTTP_200_OK)
def delete_user(user_id: int, db: Session = Depends(get_db)) -> DeleteUserResponse:
    """
    Delete a user and ALL their details:
      - InterviewReports where user_id == {user_id} OR user_cv_id in user's CVs
      - UserCVs (plus education, skills, projects, work_experience via cascade)
      - The User itself
    """
    result = delete_user_and_details(db, user_id=user_id)

    if not result.get("deleted_user"):
        # user not found -> 404
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found.",
        )

    return DeleteUserResponse(**result)
