from pydantic import BaseModel, Field
from typing import List, Optional

class GetProfileSchema(BaseModel):
    """Retrieves basic profile information, bio, and general experience."""
    pass

class ListProjectsSchema(BaseModel):
    """Searches the portfolio for projects matching optional filters."""
    filters: Optional[List[str]] = Field(
        default=None, 
        description="A list of technical tags or keywords to filter projects by (e.g., ['React', 'Python', 'AI'])"
    )

class ExplainProjectSchema(BaseModel):
    """Retrieves in-depth technical details, architecture, and reasoning for a specific project."""
    name: str = Field(
        ..., 
        description="The exact name of the project to explain (e.g., 'UptimeGuard', 'GoPlanIt')"
    )

class GetAvailabilitySchema(BaseModel):
    """Retrieves the developer's current focus, employment status, and availability for roles."""
    pass

class AnalyzeJobFitSchema(BaseModel):
    """Analyzes a job description to determine how well it matches the developer's skills and experience."""
    job_text: str = Field(
        ..., 
        description="The raw text of the job description or role requirements"
    )
