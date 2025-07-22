from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from app.models.enums import UserRole

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    phone_number: Optional[str]
    password: str
    role: UserRole = UserRole.CANDIDATE


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    phone_number: Optional[str]
    role: UserRole
    created_at: datetime

    class Config:
        from_attributes = True
