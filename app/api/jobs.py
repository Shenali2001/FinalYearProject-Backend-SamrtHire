# app/api/jobs.py (same file where your router lives)
from fastapi import APIRouter, Depends, Path, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.job import JobTypeCreate, JobTypeRead, JobPositionCreate, JobPositionRead
from app.services.job_service import (
    create_job_type_service,
    get_all_job_types_service,
    create_job_position_service,
    get_all_job_positions_service,
    get_positions_by_job_type_service,
    update_job_type_service,
    delete_job_type_service,
    update_job_position_service,
    delete_job_position_service,
)
from app.auth.dependencies import role_required

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.post("/types", response_model=JobTypeRead)
def create_job_type(job_type: JobTypeCreate, db: Session = Depends(get_db), user=Depends(role_required("admin", "recruiter"))):
    return create_job_type_service(job_type, db)

@router.get("/types", response_model=list[JobTypeRead])
def get_job_types(db: Session = Depends(get_db)):
    return get_all_job_types_service(db)

@router.put("/types/{job_type_id}", response_model=JobTypeRead)
def update_job_type(
    job_type_id: int = Path(..., description="The ID of the job type"),
    job_type: JobTypeCreate = ...,
    db: Session = Depends(get_db),
    user=Depends(role_required("admin", "recruiter")),
):
    return update_job_type_service(job_type_id, job_type, db)

@router.delete("/types/{job_type_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job_type(
    job_type_id: int = Path(..., description="The ID of the job type"),
    db: Session = Depends(get_db),
    user=Depends(role_required("admin", "recruiter")),
):
    delete_job_type_service(job_type_id, db)
    return  # 204 No Content

@router.post("/positions", response_model=JobPositionRead)
def create_job_position(position: JobPositionCreate, db: Session = Depends(get_db), user=Depends(role_required("admin", "recruiter"))):
    return create_job_position_service(position, db)

@router.get("/positions", response_model=list[JobPositionRead])
def get_job_positions(db: Session = Depends(get_db)):
    return get_all_job_positions_service(db)

@router.put("/positions/{position_id}", response_model=JobPositionRead)
def update_job_position(
    position_id: int = Path(..., description="The ID of the job position"),
    position: JobPositionCreate = ...,
    db: Session = Depends(get_db),
    user=Depends(role_required("admin", "recruiter")),
):
    return update_job_position_service(position_id, position, db)

@router.delete("/positions/{position_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job_position(
    position_id: int = Path(..., description="The ID of the job position"),
    db: Session = Depends(get_db),
    user=Depends(role_required("admin", "recruiter")),
):
    delete_job_position_service(position_id, db)
    return  # 204 No Content

@router.get("/types/{job_type_id}/positions", response_model=list[JobPositionRead])
def get_positions_by_job_type(
    job_type_id: int = Path(..., description="The ID of the job type"),
    db: Session = Depends(get_db)
):
    return get_positions_by_job_type_service(job_type_id, db)
