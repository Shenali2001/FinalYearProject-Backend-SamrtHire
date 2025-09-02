from typing import Optional, List, Dict, Any
from pydantic import BaseModel

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
    class Config:
        orm_mode = True
