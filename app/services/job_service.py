# app/services/job_service.py
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.models.job import JobType, JobPosition
from app.schemas.job import JobTypeCreate, JobPositionCreate

def create_job_type_service(job_type: JobTypeCreate, db: Session):
    new_type = JobType(name=job_type.name)
    db.add(new_type)
    db.commit()
    db.refresh(new_type)
    return new_type

def get_all_job_types_service(db: Session):
    return db.query(JobType).all()

def update_job_type_service(job_type_id: int, job_type: JobTypeCreate, db: Session):
    existing = db.query(JobType).filter(JobType.id == job_type_id).first()
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job type not found")
    existing.name = job_type.name
    db.commit()
    db.refresh(existing)
    return existing

def delete_job_type_service(job_type_id: int, db: Session):
    existing = db.query(JobType).filter(JobType.id == job_type_id).first()
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job type not found")

    # Block delete if positions still reference this type (safe default).
    has_positions = db.query(JobPosition).filter(JobPosition.type_id == job_type_id).first()
    if has_positions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete job type with existing positions. Delete or reassign positions first."
        )

    db.delete(existing)
    db.commit()

def create_job_position_service(position: JobPositionCreate, db: Session):
    new_pos = JobPosition(**position.dict())
    db.add(new_pos)
    db.commit()
    db.refresh(new_pos)
    return new_pos

def get_all_job_positions_service(db: Session):
    return db.query(JobPosition).all()

def update_job_position_service(position_id: int, position: JobPositionCreate, db: Session):
    existing = db.query(JobPosition).filter(JobPosition.id == position_id).first()
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job position not found")

    # Update all fields coming from the create schema (full PUT)
    for k, v in position.dict().items():
        setattr(existing, k, v)

    db.commit()
    db.refresh(existing)
    return existing

def delete_job_position_service(position_id: int, db: Session):
    existing = db.query(JobPosition).filter(JobPosition.id == position_id).first()
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job position not found")
    db.delete(existing)
    db.commit()

def get_positions_by_job_type_service(job_type_id: int, db: Session):
    return db.query(JobPosition).filter(JobPosition.type_id == job_type_id).all()
