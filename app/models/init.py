# Import ALL models once so they register on Base.metadata
from .user import User
from .cv import UserCV, Education, Skill, Project, WorkExperience
from .interview_reports import InterviewReport
# If you have job types/positions model files, import them too:
# from .job import JobType, JobPosition

__all__ = [
    "User",
    "UserCV", "Education", "Skill", "Project", "WorkExperience",
    "InterviewReport",
    # "JobType", "JobPosition",
]
