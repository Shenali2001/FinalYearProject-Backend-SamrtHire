# interview_reports.py
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB as JSONType
from app.database import Base
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

# SQLAlchemy Model
class InterviewReport(Base):
    __tablename__ = "interview_reports"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    user_cv_id = Column(Integer, ForeignKey("user_cvs.id", ondelete="SET NULL"), index=True, nullable=True)

    email = Column(String, nullable=False, index=True)
    role_type = Column(String)
    role_position = Column(String)

    status = Column(String, nullable=False, default="finished")
    score = Column(Integer, nullable=False, default=0)
    questions_asked = Column(Integer, nullable=False, default=0)
    accuracy_pct = Column(Float, nullable=False, default=0.0)
    suitability = Column(String)
    is_suitable = Column(Boolean, nullable=False, default=False)

    summary = Column(Text)

    strengths = Column(JSONType)
    areas_to_improve = Column(JSONType)
    next_steps = Column(JSONType)
    scorecard = Column(JSONType)
    history = Column(JSONType)
    raw_feedback = Column(JSONType)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", backref="interview_reports")
    user_cv = relationship("UserCV", backref="interview_reports")

# Pydantic Models
class InterviewReportBase(BaseModel):
    email: str
    user_id: Optional[int] = None
    user_cv_id: Optional[int] = None
    role_type: Optional[str] = None
    role_position: Optional[str] = None
    status: Optional[str] = "finished"
    score: int = 0
    questions_asked: int = 0
    accuracy_pct: float = 0.0
    suitability: Optional[str] = None
    is_suitable: bool = False
    summary: Optional[str] = None
    strengths: Optional[List[str]] = None
    areas_to_improve: Optional[List[str]] = None
    next_steps: Optional[List[str]] = None
    scorecard: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    raw_feedback: Optional[Dict[str, Any]] = None

class InterviewReportCreate(InterviewReportBase):
    pass

class InterviewReportRead(InterviewReportBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True
