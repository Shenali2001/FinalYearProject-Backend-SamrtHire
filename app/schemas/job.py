from pydantic import BaseModel
from typing import List, Optional

class JobTypeBase(BaseModel):
    name: str

class JobTypeCreate(JobTypeBase):
    pass

class JobTypeRead(JobTypeBase):
    id: int

    class Config:
        orm_mode = True

class JobPositionBase(BaseModel):
    name: str
    type_id: int

class JobPositionCreate(JobPositionBase):
    pass

class JobPositionRead(JobPositionBase):
    id: int

    class Config:
        orm_mode = True

class JobPositionRead(BaseModel):
    id: int
    name: str
    type_id: int

    class Config:
        orm_mode = True
