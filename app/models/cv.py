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

# Helper function to validate CV against template
def validate_cv_template(text: str) -> bool:
    """
    Validate that the CV contains all required section headers.

    Args:
        text (str): The extracted text from the CV.

    Returns:
        bool: True if the CV matches the template, False otherwise.
    """
    required_sections = ["SUMMARY:", "EDUCATION:", "SKILLS:", "PROJECTS:", "WORK EXPERIENCE:"]
    text_upper = text.upper()
    missing_sections = [section for section in required_sections if section not in text_upper]
    if missing_sections:
        logger.warning(f"Missing sections: {missing_sections}")
        return False
    logger.debug("CV template validated successfully")
    return True
# Helper function to extract section text
def extract_section_text(text: str, section_pattern: str, next_patterns: list) -> str:
    """
    Extract the text for a specific section between its header and the next section header.

    Args:
        text (str): The full extracted text from the PDF.
        section_pattern (str): Regex pattern for the section header.
        next_patterns (list): List of regex patterns for possible next section headers.

    Returns:
        str: The extracted section text, or empty string if not found.
    """
    match = re.search(section_pattern + r"\s*(?:\n|$)", text, re.IGNORECASE)
    if not match:
        logger.warning(f"Section not found for pattern: {section_pattern}")
        return ""

    start = match.end()
    end = len(text)
    for pattern in next_patterns:
        next_match = re.search(pattern + r"\s*(?:\n|$)", text[start:], re.IGNORECASE)
        if next_match:
            end = min(end, start + next_match.start())

    section_text = text[start:end].strip()
    logger.debug(f"Extracted section text for {section_pattern}: {section_text[:100]}...")
    return section_text