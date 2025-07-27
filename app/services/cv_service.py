import requests
import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models.cv import UserCV, Education, Skill, Project, WorkExperience
from app.models.user import User
from app.schemas.cv_details import UserCVCreate, EducationCreate, SkillCreate, ProjectCreate, WorkExperienceCreate
import fitz  # PyMuPDF
import re
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper function to extract text from PDF
def extract_text_from_pdf_url(pdf_url: str) -> str:
    """
    Download PDF from URL, save it locally as temp_cv.pdf, and extract all text using PyMuPDF.

    Args:
        pdf_url (str): URL of the PDF to extract text from.

    Returns:
        str: Extracted text from the PDF.

    Raises:
        Exception: If the PDF cannot be downloaded or processed.
    """
    logger.debug(f"Downloading PDF from URL: {pdf_url}")
    response = requests.get(pdf_url)
    if response.status_code != 200:
        logger.error(f"Failed to download CV from URL: {pdf_url}, status code: {response.status_code}")
        raise Exception("Failed to download CV from provided URL")

    local_pdf_path = "temp_cv.pdf"
    # with open(local_pdf_path, "wb") as f:
    #     f.write(response.content)

    try:
        doc = fitz.open(local_pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")
    finally:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
            logger.debug(f"Removed temporary file: {local_pdf_path}")

    # Normalize only multiple spaces/tabs, preserve newlines
    text = re.sub(r'[ \t]+', ' ', text.strip())
    logger.debug(f"Extracted text (first 500 chars): {text[:500]}...")
    logger.debug(f"Full extracted text:\n{text}")
    return text

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

# Helper function to parse education section
def parse_education_section(section_text: str) -> list:
    """
    Parse the education section text into a list of education entries.

    Args:
        section_text (str): The text of the education section.

    Returns:
        list: List of dictionaries containing education data.
    """
    education = []
    if not section_text:
        logger.warning("Education section is empty")
        return education

    lines = section_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•'):
            parts = line.strip('-• ').split(',', 2)
            if len(parts) >= 3:
                institution_degree = parts[0].strip() + ", " + parts[1].strip()
                years_notes = parts[2].strip()
                years = re.search(r'(\d{4})\s*-\s*(\d{4}|Present)', years_notes)
                if years:
                    edu_data = {
                        "institution": institution_degree.split(',')[0].strip(),
                        "degree": institution_degree.split(',')[1].strip(),
                        "start_year": years.group(1),
                        "end_year": years.group(2),
                        "notes": years_notes[years.end():].strip() if years.end() < len(years_notes) else ""
                    }
                    try:
                        EducationCreate(**edu_data)
                        education.append(edu_data)
                        logger.debug(f"Parsed education: {edu_data}")
                    except ValueError as e:
                        logger.warning(f"Invalid education data: {edu_data}, error: {str(e)}")
    return education

# Helper function to parse skills section
def parse_skills_section(section_text: str) -> list:
    """
    Parse the skills section text into a list of skill entries.

    Args:
        section_text (str): The text of the skills section.

    Returns:
        list: List of dictionaries containing skill data.
    """
    skills = []
    if not section_text:
        logger.warning("Skills section is empty")
        return skills

    lines = section_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•'):
            skill_list = [s.strip() for s in line.strip('-• ').split(',') if s.strip()]
            for skill in skill_list:
                skill_data = {"skill_name": skill}
                try:
                    SkillCreate(**skill_data)
                    skills.append(skill_data)
                    logger.debug(f"Parsed skill: {skill_data}")
                except ValueError as e:
                    logger.warning(f"Invalid skill data: {skill_data}, error: {str(e)}")
    return skills

# Helper function to parse projects section
def parse_projects_section(section_text: str) -> list:
    """
    Parse the projects section text into a list of project entries.

    Args:
        section_text (str): The text of the projects section.

    Returns:
        list: List of dictionaries containing project data.
    """
    projects = []
    if not section_text:
        logger.warning("Projects section is empty")
        return projects

    lines = section_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•'):
            line = line.strip('-• ').strip()
            if ': ' in line:
                name_tech = line.split(': ', 1)
                name = name_tech[0].strip()
                tech_desc = name_tech[1].split(', Technologies: ', 1)
                description = tech_desc[0].strip()
                technologies = tech_desc[1].strip() if len(tech_desc) > 1 else ""
                proj_data = {
                    "name": name,
                    "description": description,
                    "technologies": technologies
                }
                try:
                    ProjectCreate(**proj_data)
                    projects.append(proj_data)
                    logger.debug(f"Parsed project: {proj_data}")
                except ValueError as e:
                    logger.warning(f"Invalid project data: {proj_data}, error: {str(e)}")
    return projects

# Helper function to parse work experience section
def parse_work_experience_section(section_text: str) -> list:
    """
    Parse the work experience section text into a list of work experience entries.

    Args:
        section_text (str): The text of the work experience section.

    Returns:
        list: List of dictionaries containing work experience data.
    """
    work_experience = []
    if not section_text:
        logger.warning("Work experience section is empty")
        return work_experience

    lines = section_text.split('\n')
    work = {}
    description_lines = []
    for line in lines:
        line = line.strip()
        if (line.startswith('-') or line.startswith('•')) and ',' in line:
            if work:
                work["description"] = " ".join(description_lines).strip()
                try:
                    WorkExperienceCreate(**work)
                    work_experience.append(work)
                    logger.debug(f"Parsed work experience: {work}")
                except ValueError as e:
                    logger.warning(f"Invalid work experience data: {work}, error: {str(e)}")
                description_lines = []
            parts = line.strip('-• ').split(',', 2)
            if len(parts) >= 3:
                company = parts[0].strip()
                role = parts[1].strip()
                years_desc = parts[2].strip()
                years = re.search(r'(\d{4})\s*-\s*(\d{4}|Present)', years_desc)
                if years:
                    work = {
                        "company": company,
                        "role": role,
                        "start_date": years.group(1),
                        "end_date": years.group(2),
                        "description": ""
                    }
        elif (line.startswith('-') or line.startswith('•')) and work:
            description_lines.append(line.strip('-• ').strip())
    if work:
        work["description"] = " ".join(description_lines).strip()
        try:
            WorkExperienceCreate(**work)
            work_experience.append(work)
            logger.debug(f"Parsed work experience: {work}")
        except ValueError as e:
            logger.warning(f"Invalid work experience data: {work}, error: {str(e)}")
    return work_experience

# Main parsing function
def parse_cv_text(text: str) -> dict:
    """
    Parse the CV text into structured data for education, skills, projects, and work experience.

    Args:
        text (str): The extracted text from the CV PDF.

    Returns:
        dict: A dictionary with keys 'education', 'skills', 'projects', and 'work_experience',
              each containing a list of dictionaries with relevant fields.

    Raises:
        ValueError: If the CV does not match the required template or any section is empty.
    """
    if not validate_cv_template(text):
        logger.error("CV does not match the required template")
        raise ValueError("CV does not match the required template. Please upload a CV with the sections: SUMMARY:, EDUCATION:, SKILLS:, PROJECTS:, WORK EXPERIENCE:")

    section_headers = {
        "summary": r"SUMMARY:",
        "education": r"EDUCATION:",
        "skills": r"SKILLS:",
        "projects": r"PROJECTS:",
        "work_experience": r"WORK EXPERIENCE:"
    }
    all_patterns = list(section_headers.values())

    parsed = {
        "education": parse_education_section(
            extract_section_text(text, section_headers["education"], all_patterns[2:])),
        "skills": parse_skills_section(extract_section_text(text, section_headers["skills"], all_patterns[3:])),
        "projects": parse_projects_section(extract_section_text(text, section_headers["projects"], all_patterns[4:])),
        "work_experience": parse_work_experience_section(
            extract_section_text(text, section_headers["work_experience"], all_patterns[5:]))
    }

    # Check if any section is empty
    empty_sections = [key for key, value in parsed.items() if not value]
    if empty_sections:
        logger.error(f"Empty sections detected: {empty_sections}")
        raise ValueError(f"CV sections {empty_sections} are empty. Please upload a CV with complete sections: SUMMARY:, EDUCATION:, SKILLS:, PROJECTS:, WORK EXPERIENCE:")

    logger.debug(f"Parsed CV data: {parsed}")
    return parsed

# Service function to create UserCV record
def create_user_cv_service(user_email: str, cv_data: UserCVCreate, db: Session):
    """
    Create a UserCV record from a CV PDF and store structured data in the database.

    Args:
        user_email (str): Email of the user.
        cv_data (UserCVCreate): Schema containing CV URL and other metadata.
        db (Session): Database session.

    Returns:
        UserCV: The created UserCV object.

    Raises:
        HTTPException: If user not found, CV processing fails, or CV does not match template.
    """
    logger.debug(f"Creating UserCV for user: {user_email}, cv_data: {cv_data}")
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        logger.error(f"User not found: {user_email}")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        cv_text = extract_text_from_pdf_url(cv_data.cv_url)
        parsed_data = parse_cv_text(cv_text)
    except ValueError as ve:
        logger.error(f"CV parsing failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"CV processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"CV processing failed: {str(e)}")

    user_cv = UserCV(
        user_id=user.id,
        cv_url=cv_data.cv_url,
        job_type_id=cv_data.job_type_id,
        job_position_id=cv_data.job_position_id,
        extracted_text=cv_text
    )
    logger.debug(f"Created UserCV object: {user_cv.__dict__}")

    for edu_data in parsed_data.get("education", []):
        try:
            edu = EducationCreate(**edu_data)
            education = Education(**edu.dict())
            user_cv.education.append(education)
            logger.debug(f"Appended education: {education.__dict__}")
        except ValueError as e:
            logger.warning(f"Failed to validate education data: {edu_data}, error: {str(e)}")

    for skill_data in parsed_data.get("skills", []):
        try:
            skill = SkillCreate(**skill_data)
            skill_obj = Skill(**skill.dict())
            user_cv.skills.append(skill_obj)
            logger.debug(f"Appended skill: {skill_obj.__dict__}")
        except ValueError as e:
            logger.warning(f"Failed to validate skill data: {skill_data}, error: {str(e)}")

    for proj_data in parsed_data.get("projects", []):
        try:
            proj = ProjectCreate(**proj_data)
            project = Project(**proj.dict())
            user_cv.projects.append(project)
            logger.debug(f"Appended project: {project.__dict__}")
        except ValueError as e:
            logger.warning(f"Failed to validate project data: {proj_data}, error: {str(e)}")

    for work_data in parsed_data.get("work_experience", []):
        try:
            work = WorkExperienceCreate(**work_data)
            work_exp = WorkExperience(**work.dict())
            user_cv.work_experience.append(work_exp)
            logger.debug(f"Appended work experience: {work_exp.__dict__}")
        except ValueError as e:
            logger.warning(f"Failed to validate work experience data: {work_data}, error: {str(e)}")

    db.add(user_cv)
    db.commit()
    db.refresh(user_cv)
    logger.debug(f"Saved UserCV with id: {user_cv.id}")
    return user_cv

# Service function to retrieve all UserCV records
def get_all_user_cvs_service(db: Session):
    """
    Retrieve all UserCV records from the database.

    Args:
        db (Session): Database session.

    Returns:
        List[UserCV]: List of all UserCV objects.
    """
    cvs = db.query(UserCV).all()
    logger.debug(f"Retrieved {len(cvs)} UserCV records")
    return cvs