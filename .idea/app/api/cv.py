from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.cv import UserCVCreate, UserCVRead
from app.services.cv_service import (
    create_user_cv_service,
    get_all_user_cvs_service,
)
from app.auth.dependencies import role_required
from pydantic import BaseModel
import logging


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cv", tags=["User CV"])

# Schema for evaluate_answer endpoint
class AnswerEvaluationRequest(BaseModel):
    question: str
    answer: str

@router.post("/", response_model=UserCVRead)
def upload_cv(
    cv_data: UserCVCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(role_required("candidate"))
):
    logger.debug(f"Uploading CV for user: {current_user['email']}")
    return create_user_cv_service(current_user["email"], cv_data, db)


@router.get("/", response_model=list[UserCVRead])
def get_all_user_cvs(
    db: Session = Depends(get_db),
    _: dict = Depends(role_required("admin"))
):
    logger.debug("Retrieving all CVs for admin")
    return get_all_user_cvs_service(db)


