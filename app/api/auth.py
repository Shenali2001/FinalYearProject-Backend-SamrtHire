from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut, UserLogin
from app.services.auth_service import register_user, authenticate_user
from app.core.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.database import get_db
from datetime import timedelta

router = APIRouter()

@router.post("/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    return register_user(user, db)

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = authenticate_user(user.email, user.password, db)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Include the user's role in the token
    access_token = create_access_token(
        subject=db_user.email,
        role=db_user.role.value,  # use .value if role is an Enum
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": db_user.email,
            "role": db_user.role.value
        }
    }