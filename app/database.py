from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/ai_interview")

# Engine and Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# Import all models to register them with Base
from .models import cv, interview_reports, job, user  # Adjust based on your model files

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()