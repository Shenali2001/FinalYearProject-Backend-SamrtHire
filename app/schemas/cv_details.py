from pydantic import BaseModel, Field
from typing import Optional, List

class EducationBase(BaseModel):
    institution: str
    degree: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    notes: Optional[str] = None

class EducationCreate(EducationBase):
    pass

class EducationRead(EducationBase):
    id: int
    user_cv_id: int

    class Config:
        from_attributes = True

class SkillBase(BaseModel):
    skill_name: str

class SkillCreate(SkillBase):
    pass

class SkillRead(SkillBase):
    id: int
    user_cv_id: int

    class Config:
        from_attributes = True

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectRead(ProjectBase):
    id: int
    user_cv_id: int

    class Config:
        from_attributes = True

class WorkExperienceBase(BaseModel):
    company: str
    role: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class WorkExperienceCreate(WorkExperienceBase):
    pass

class WorkExperienceRead(WorkExperienceBase):
    id: int
    user_cv_id: int

    class Config:
        from_attributes = True

class UserCVCreate(BaseModel):
    cv_url: str
    job_type_id: int
    job_position_id: int

class UserCVRead(BaseModel):
    id: int
    user_id: int
    cv_url: str
    job_type_id: int
    job_position_id: int
    extracted_text: Optional[str] = None
    education: List[EducationRead] = Field(default_factory=list)
    skills: List[SkillRead] = Field(default_factory=list)
    projects: List[ProjectRead] = Field(default_factory=list)
    work_experience: List[WorkExperienceRead] = Field(default_factory=list)

    class Config:
        from_attributes = True