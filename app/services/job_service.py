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

def create_job_position_service(position: JobPositionCreate, db: Session):
    new_pos = JobPosition(**position.dict())
    db.add(new_pos)
    db.commit()
    db.refresh(new_pos)
    return new_pos

def get_all_job_positions_service(db: Session):
    return db.query(JobPosition).all()

def get_positions_by_job_type_service(job_type_id: int, db: Session):
    return db.query(JobPosition).filter(JobPosition.type_id == job_type_id).all()
