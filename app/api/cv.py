from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.cv import UserCVCreate, UserCVRead
from app.services.cv_service import create_user_cv_service, get_all_user_cvs_service
from app.auth.dependencies import role_required
from app.models.user import User

router = APIRouter(prefix="/cv", tags=["User CV"])

@router.post("/", response_model=UserCVRead)
def upload_cv(
    cv_data: UserCVCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(role_required("candidate"))
):
    return create_user_cv_service(current_user["email"], cv_data, db)

@router.get("/", response_model=list[UserCVRead])
def get_all_user_cvs(
    db: Session = Depends(get_db),
    _: User = Depends(role_required("admin"))
):
    return get_all_user_cvs_service(db)