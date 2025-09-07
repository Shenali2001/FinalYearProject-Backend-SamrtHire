from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class UserCV(Base):
    __tablename__ = "user_cvs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    cv_url = Column(String, nullable=False)
    job_type_id = Column(Integer, ForeignKey("job_types.id"))
    job_position_id = Column(Integer, ForeignKey("job_positions.id"))
    extracted_text = Column(Text)

    user = relationship("User", back_populates="cvs")
    job_type = relationship("JobType")
    job_position = relationship("JobPosition")

    education = relationship("Education", back_populates="user_cv", cascade="all, delete-orphan")
    skills = relationship("Skill", back_populates="user_cv", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="user_cv", cascade="all, delete-orphan")
    work_experience = relationship("WorkExperience", back_populates="user_cv", cascade="all, delete-orphan")

class Education(Base):
    __tablename__ = "education"

    id = Column(Integer, primary_key=True, index=True)
    user_cv_id = Column(Integer, ForeignKey("user_cvs.id"))
    institution = Column(String)
    degree = Column(String)
    start_year = Column(String)
    end_year = Column(String)
    notes = Column(Text)

    user_cv = relationship("UserCV", back_populates="education")

class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, index=True)
    user_cv_id = Column(Integer, ForeignKey("user_cvs.id"))
    skill_name = Column(String)

    user_cv = relationship("UserCV", back_populates="skills")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    user_cv_id = Column(Integer, ForeignKey("user_cvs.id"))
    name = Column(String)
    description = Column(Text)
    technologies = Column(Text)

    user_cv = relationship("UserCV", back_populates="projects")

class WorkExperience(Base):
    __tablename__ = "work_experience"

    id = Column(Integer, primary_key=True, index=True)
    user_cv_id = Column(Integer, ForeignKey("user_cvs.id"))
    company = Column(String)
    role = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    description = Column(Text)

    user_cv = relationship("UserCV", back_populates="work_experience")