from pydantic import BaseModel

class UserCVBase(BaseModel):
    cv_url: str
    job_type_id: int
    job_position_id: int

class UserCVCreate(UserCVBase):
    pass

class UserCVRead(UserCVBase):
    id: int
    user_id: int
    extracted_text: str

    class Config:
        orm_mode = True
