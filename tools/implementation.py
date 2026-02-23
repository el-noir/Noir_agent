from typing import List, Optional, Dict, Any

# Static mock data representing the structured portfolio
PROFILE_DATA = {
    "name": "Mudasir Shah",
    "title": "Full-Stack Developer & AI Specialist",
    "bio": "Started my journey in December 2023 with backend development. Evolved into full-stack development with React and Next.js, then advanced to building AI-powered applications using LangChain and LangGraph.",
    "skills": {
        "languages": ["JavaScript (ES6+)", "TypeScript", "Python", "C++", "SQL"],
        "frontend": ["React", "Next.js", "Tailwind CSS"],
        "backend": ["Node.js", "NestJS", "FastAPI", "Express"],
        "ai": ["LangChain", "LangGraph", "RAG"]
    }
}

PROJECTS_DATA = [
    {
        "name": "UptimeGuard",
        "description": "Decentralized uptime monitoring with crypto-verified validators.",
        "tags": ["React", "Express", "Node.js", "PostgreSQL", "WebSockets", "Prisma"],
        "architecture_notes": "Built tracking website status in real-time with historical analytics."
    },
    {
        "name": "GoPlanIt",
        "description": "AI travel itineraries using real-time flight and attraction data.",
        "tags": ["React", "Node.js", "Express", "Gemini API", "Inngest", "MongoDB"],
        "architecture_notes": "Delivered tailored recommendations simplifying travel planning. Handled async processing via Inngest."
    },
    {
        "name": "Airgpt",
        "description": "RAG system built for Air University.",
        "tags": ["Next.js", "React", "FastAPI", "Python", "Qdrant", "LangChain"],
        "architecture_notes": "Answers user queries by searching and reasoning over ingested documents using a vector database (Qdrant)."
    }
]

def getProfile() -> Dict[str, Any]:
    """Retrieves basic profile information."""
    return PROFILE_DATA

def listProjects(filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Returns a list of projects, optionally filtered by tags."""
    if not filters:
        return [{"name": p["name"], "description": p["description"], "tags": p["tags"]} for p in PROJECTS_DATA]
    
    # Case-insensitive filtering
    filter_lower = [f.lower() for f in filters]
    results = []
    for p in PROJECTS_DATA:
        tags_lower = [t.lower() for t in p["tags"]]
        if any(f in tags_lower for f in filter_lower):
            results.append({"name": p["name"], "description": p["description"], "tags": p["tags"]})
            
    return results

def explainProject(name: str) -> Dict[str, Any]:
    """Provides deep architectural details for a specific project."""
    for p in PROJECTS_DATA:
        if p["name"].lower() == name.lower():
            return {
                "name": p["name"],
                "architecture_notes": p["architecture_notes"],
                "tags": p["tags"]
            }
    return {"error": f"Project '{name}' not found in portfolio."}

def getAvailability() -> Dict[str, str]:
    """Returns current working status and preferred roles."""
    return {
        "status": "Currently employed as Web Developer at Out-Secure (Oct 2024 - Present).",
        "open_to": "Exploring opportunities involving AI infrastructure, advanced backend systems, or lead full-stack roles."
    }

def analyzeJobFit(job_text: str) -> Dict[str, Any]:
    """Basic mock implementation representing fit extraction."""
    # In a full app, this would use a lightweight LLM pass to compare `job_text` with `PROFILE_DATA`.
    # For now, we mock the structured response.
    return {
        "score": "Strong Fit (estimated)",
        "matching_skills": ["Backend APIs", "Full-Stack Development"],
        "missing_skills": ["Specific proprietary enterprise tools (e.g., Salesforce)"],
        "analysis": "The developer has strong foundational backend and AI skills matching typical modern SaaS roles."
    }
