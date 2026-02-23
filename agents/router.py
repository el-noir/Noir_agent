from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

# Using a hardcoded model for structure; can be parameterized later
try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    # Fallback if no API key is set yet
    llm = None

class RouteStrategy(BaseModel):
    """Routing schema for the user query."""
    intent: Literal["ABOUT", "PROJECT", "TECH_DEEP_DIVE", "AVAILABILITY", "COLLABORATION", "OUT_OF_SCOPE"] = Field(
        ...,
        description="Given a user question, choose to route it to the appropriate specialized agent."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a master routing agent for a top-tier software engineer's portfolio AI.
    Your job is to analyze the user's intent and route them to the correct specialized agent.
    
    Choose ONE of the following intents:
    - ABOUT: Questions about the engineer's bio, soft skills, history, or resume.
    - PROJECT: Broad questions about projects they've worked on.
    - TECH_DEEP_DIVE: Highly specific technical questions (e.g., "Explain the network scanner", "What is the architecture of X?").
    - AVAILABILITY: Questions about working together, hiring, or scheduling.
    - COLLABORATION: Questions about teamwork or side projects.
    - OUT_OF_SCOPE: Questions completely unrelated to software engineering or the portfolio (e.g., "What is the capital of France?").
    """),
    ("human", "{question}")
])

def router_agent(state: dict):
    """
    Analyzes the user's query and determines the intent.
    """
    print("---ROUTE QUERY---")
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Check if LLM is properly initialized
    if not llm:
        print("Warning: LLM not initialized. Defaulting to ABOUT.")
        return {"intent": "ABOUT", "metadata": {"routed_to": "portfolio"}}
    
    structured_llm_router = llm.with_structured_output(RouteStrategy)
    router_chain = router_prompt | structured_llm_router
    
    result = router_chain.invoke({"question": last_message})
    intent = result.intent
    print(f"---INTENT ASSESSED: {intent}---")
    
    # Store intent in state
    return {"intent": intent, "metadata": {"routed_to": intent}}

def route_query(state: dict):
    """
    Edge function that reads the intent and dictates the next node.
    """
    intent = state.get("intent", "ABOUT")
    
    # Map intents to nodes
    if intent in ["ABOUT", "AVAILABILITY", "COLLABORATION"]:
        return "ABOUT" # Maps to "portfolio" node in graph.py
    elif intent in ["PROJECT", "TECH_DEEP_DIVE"]:
        return "PROJECT" # Maps to "project" node
    elif intent == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    else:
        return "DEFAULT"
