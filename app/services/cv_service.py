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