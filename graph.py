import os
import json
import time
from typing import Annotated, TypedDict, Union, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from tools.implementation import (
    getProfile, listProjects, explainProject, getAvailability, analyzeJobFit
)
from calendar_mcp import get_calendar_tools

load_dotenv()

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    trace: Dict[str, Any]
    active_agent: str

# --- LLM Initialization ---
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# --- Portfolio Tools ---
from langchain_core.tools import tool

@tool
def get_profile_tool() -> str:
    """Retrieves Mudasir's basic profile information, bio, and general experience."""
    return json.dumps(getProfile())

@tool
def list_projects_tool(filters: str = None) -> str:
    """Searches the portfolio for projects matching optional filters like 'React, Python, AI'. Pass a comma-separated string if any filters apply."""
    filter_list = [f.strip() for f in filters.split(',')] if filters else None
    return json.dumps(listProjects(filter_list))

@tool
def explain_project_tool(name: str) -> str:
    """Retrieves in-depth technical details, architecture, and reasoning for a specific project by exact name."""
    return json.dumps(explainProject(name))

@tool
def get_availability_tool() -> str:
    """Retrieves the developer's current focus, employment status, and availability for roles."""
    return json.dumps(getAvailability())

@tool
def analyze_job_fit_tool(job_text: str) -> str:
    """Analyzes a job description to determine how well it matches Mudasir's skills and experience."""
    return json.dumps(analyzeJobFit(job_text))

portfolio_tools = [
    get_profile_tool,
    list_projects_tool,
    explain_project_tool,
    get_availability_tool,
    analyze_job_fit_tool
]

# --- Calendar Tools ---
from calendar_mcp import get_calendar_tools

# --- Nodes ---

def router_node(state: AgentState):
    """
    Decides whether to route to Portfolio or Calendar agent.
    """
    user_input = state["messages"][-1].content
    
    system_prompt = (
        "Analyze the user's message and decide if it's related to Mudasir's PORTFOLIO (projects, skills, bio, job fit) "
        "or his CALENDAR/SCHEDULING (meetings, what he's doing today, booking time). "
        "Respond with exactly one word: 'portfolio' or 'calendar'."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    decision = response.content.lower().strip()
    if 'calendar' in decision:
        return {"active_agent": "calendar"}
    return {"active_agent": "portfolio"}

def portfolio_chatbot(state: AgentState):
    """Handles portfolio-related inquiries."""
    llm_with_tools = llm.bind_tools(portfolio_tools)
    
    system_prompt = (
        "You are Mudasir Shah's AI Assistant, 'Noir AI'. "
        "Use the provided tools to answer questions about his experience, projects, and skills.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. BE EXTREMELY CONCISE. If the user asks a basic question like 'Who is Mudasir Shah?', provide a short 1-2 sentence answer (e.g. 'Mudasir is a Full-Stack Developer and AI Specialist.'). DO NOT list all his skills or history unless explicitly asked.\n"
        "2. Speak confidently and directly. NEVER use robotic filler phrases like 'It appears that', 'It seems that', or 'Based on the provided context'. Answer directly as a knowledgeable AI.\n"
        "3. Only use bullet points or detailed lists if the user specifically asks for 'details', 'everything', 'list', or 'tell me more'."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Update trace
    trace = state.get("trace", {})
    trace["active_agent"] = "Portfolio Agent"
    if response.tool_calls:
        trace["intent_detected"] = "Tool Execution"
        trace["tool_selected"] = response.tool_calls[0]["name"]
        trace["tool_args"] = response.tool_calls[0]["args"]
    else:
        trace["intent_detected"] = "Direct Answer"
        
    return {"messages": [response], "trace": trace}

async def calendar_chatbot(state: AgentState):
    """Handles calendar-related inquiries."""
    trace = state.get("trace", {})
    trace["active_agent"] = "Calendar Agent"
    
    # Reload tools asynchronously if needed
    tools = await get_calendar_tools()
    if not tools:
        return {
            "messages": [AIMessage(content="I'm sorry, I cannot access the calendar right now because the credentials are not set up.")],
            "trace": trace
        }
    
    llm_with_tools = llm.bind_tools(tools)
    now = datetime.now()
    current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
    
    system_prompt = (
        f"You are Mudasir Shah's AI Assistant, 'Noir AI'.\n"
        f"Current Time: {current_time}\n"
        "Manage the user's schedule. Before creating events, check for conflicts.\n"
        "IMPORTANT: When calling calendar tools, ALWAYS use the account name 'normal'.\n"
        "CRITICAL: The calendar ID for Mudasir Shah is ALWAYS 'primary'. Do not use any other calendar ID.\n"
        "TOOL JSON SCHEMA WARNING: The 'create-event' tool strictly requires arrays for complex types. NEVER pass stringified arrays (e.g., '[\"email\"]') for the 'attendees' or 'reminders.overrides' parameters. You MUST pass actual JSON arrays, or leave them out completely if not strictly necessary.\n"
        "TONE & FORMAT INSTRUCTIONS:\n"
        "1. Speak confidently and directly. NEVER use phrases like 'It appears that', 'It seems that', or 'Based on the schedule'. Answer directly.\n"
        "2. BE CONCISE. Do not add fluff. Just state the schedule."
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    
    if response.tool_calls:
        trace["intent_detected"] = "Tool Execution"
        trace["tool_selected"] = response.tool_calls[0]["name"]
        trace["tool_args"] = response.tool_calls[0]["args"]
    else:
        trace["intent_detected"] = "Direct Answer"
        
    return {"messages": [response], "trace": trace}

# --- Conditional Edges ---

def route_decision(state: AgentState):
    if state["active_agent"] == "calendar":
        return "calendar_chatbot"
    return "portfolio_chatbot"

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- Graph Construction ---

async def create_portfolio_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("portfolio_chatbot", portfolio_chatbot)
    workflow.add_node("calendar_chatbot", calendar_chatbot)
    
    # Tool nodes - Combine all tools
    cal_tools = await get_calendar_tools()
    all_tools = portfolio_tools + cal_tools
    workflow.add_node("tools", ToolNode(all_tools))
    
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", route_decision, ["portfolio_chatbot", "calendar_chatbot"])
    
    workflow.add_conditional_edges("portfolio_chatbot", should_continue, ["tools", END])
    workflow.add_conditional_edges("calendar_chatbot", should_continue, ["tools", END])
    
    # After tools run, we need to return to the active agent
    def return_to_agent(state: AgentState):
        if state["active_agent"] == "calendar":
            return "calendar_chatbot"
        return "portfolio_chatbot"
        
    workflow.add_conditional_edges("tools", return_to_agent, ["portfolio_chatbot", "calendar_chatbot"])
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
