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

@tool
def book_meeting_tool(name: str, email: str, start_time: str, end_time: str, description: str = "") -> str:
    """
    Schedules a meeting on Mudasir's calendar.
    Pass the user's explicit name, their exact email, and the start/end times in ISO 8601 format 
    (e.g., '2026-02-25T14:00:00').
    """
    # This acts as a proxy tool to shield the LLM from the complex 'attendees' array schema
    import json
    import subprocess
    import os

    # Enforce timezone offset if missing (Asia/Karachi is UTC+5)
    def add_tz(ts: str) -> str:
        if "+" not in ts and "Z" not in ts:
            return ts + "+05:00"
        return ts

    payload = {
        "account": "normal",
        "calendarId": "primary",
        "summary": f"Meeting with {name}",
        "description": description,
        "start": add_tz(start_time),
        "end": add_tz(end_time),
        "attendees": [{"email": email, "optional": False, "responseStatus": "needsAction"}],
        "timeZone": "Asia/Karachi"
    }
    
    # Path to credentials.json in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    
    if not os.path.exists(creds_path):
        return json.dumps({"status": "error", "message": "credentials.json not found. Cannot book meeting."})

    # Prepare the subprocess environment variable
    env = os.environ.copy()
    env["GOOGLE_OAUTH_CREDENTIALS"] = creds_path
    
    # In order to trigger an MCP tool natively via CLI, we can write a tiny ephemeral node script 
    # to invoke the tool, but since this is an MCP server it primarily speaks JSON-RPC over stdio.
    # To avoid rewriting an entire MCP client inside this proxy, we use LangChain's built-in MCP invocation.
    # Wait, the event loop conflict prevents us from awaiting the client here.
    
    # The simplest, most reliable way to create a Google Calendar event from Python (since we have the credentials.json)
    # is to just use the standard google-api-python-client directly instead of wrapping stdio MCP inside a synchronous tool.
    # But since the requirement is to use the MCP pattern, and LangChain prevents `async def` tools from being easily 
    # called within a synchronous graph state transition without throwing loop errors, we'll execute a tiny 
    # Python script in a subprocess to do the async MCP call in an isolated event loop.

    import tempfile
    
    isolated_script = f"""
import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    server_params = StdioServerParameters(
        command="node",
        args=[r"{os.path.join(script_dir, 'node_modules', '@cocal', 'google-calendar-mcp', 'build', 'index.js')}"],
        env={{"GOOGLE_OAUTH_CREDENTIALS": r"{creds_path}"}}
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("create-event", arguments={json.dumps(payload)})
                print(result.content[0].text)
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))

if __name__ == "__main__":
    asyncio.run(run())
"""
    
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(isolated_script)
            
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            env=env
        )
        output = result.stdout.strip()
        if not output:
             output = result.stderr.strip()
             
        return json.dumps({
            "status": "success" if result.returncode == 0 else "error",
            "mcp_output": output,
            "message": f"Routed the meeting payload for {name} ({email}) at {start_time}"
        })
    finally:
        os.remove(path)

# --- Nodes ---

def identify_intent(state: AgentState):
    """
    Decides whether to route to Portfolio or Calendar agent.
    """
    user_input = state["messages"][-1].content
    
    system_prompt = (
        "Analyze the user's message to decide the intent. Respond with exactly one word: 'portfolio' or 'calendar'.\n"
        "RULES:\n"
        "1. ESCAPE HATCH (CRITICAL): If the user says 'no', 'nope', 'nevermind', 'stop', 'cancel', or indicates they do NOT want a meeting anymore -> 'portfolio'.\n"
        "2. If the user mentions meetings, tomorrow, scheduling, booking, time, or calendar -> 'calendar'.\n"
        "3. If the user is just providing their name, email, or a time (answering a follow-up question to book a meeting) -> 'calendar'.\n"
        "4. If they ask about Mudasir's skills, projects, background, resume, or 'who is he' -> 'portfolio'."
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
        "You are 'Noir AI', Mudasir Shah's high-end personal concierge and AI Assistant. "
        "Use the provided tools to answer questions about his experience, projects, and skills.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. BE EXTREMELY CONCISE. If the user asks a basic question like 'Who is Mudasir Shah?', provide a short 1-2 sentence answer. DO NOT list all his skills or history unless explicitly asked.\n"
        "2. TONE: Speak confidently, professionally, and warmly. Answer directly as a knowledgeable AI without using robotic filler phrases like 'It appears that'.\n"
        "3. EMOJIS: Use emojis sparingly to make the conversation friendly (max 1 per message). Avoid looking like a spam bot.\n"
        "4. GUIDED INTERVIEW: When summarizing Mudasir's background or projects, end your response with a question to guide the user naturally (e.g., '...I can tell you more about his AI work or Frontend projects. Which interests you more?').\n"
        "5. SMART TRANSITIONS: If a user asks deeply technical questions and seems highly impressed, smoothly offer a meeting: 'I can go into more detail, or if you'd prefer, I can schedule a quick technical chat with Mudasir directly right now.'\n"
        "6. PROACTIVE SCHEDULING: If the user asks to talk to him or hire him, proactively tell them you can schedule a meeting right now in this chat.\n"
        "7. GRACEFUL CANCELLATION: If a user changes their mind about a meeting, say 'No problem, let me know if you change your mind!'. Do NOT keep asking.\n"
        "8. STRICT SCOPE: You MUST ONLY answer questions related to Mudasir, his projects, skills, and availability. Refuse unrelated topics (e.g., Elon Musk) and steer back to Mudasir.\n"
        "9. TOOL USAGE: ALWAYS use tools correctly. NEVER output raw tool syntax like `<function=...>` in your text."
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
    
    # Instead of forcing the LLM to learn the complex create-event schema, 
    # we give it our simple proxy tool, and keep ONLY read-only MCP tools
    tools = await get_calendar_tools()
    read_only_mcp_tools = [t for t in tools if t.name in ["list-calendars", "list-events"]]
    
    # Combine read-only MCP tools with our robust proxy tool
    calendar_agent_tools = read_only_mcp_tools + [book_meeting_tool]
    
    if not calendar_agent_tools:
        return {
            "messages": [AIMessage(content="I'm sorry, I cannot access the calendar right now because the credentials are not set up.")],
            "trace": trace
        }
    
    llm_with_tools = llm.bind_tools(calendar_agent_tools)
    now = datetime.now()
    current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
    
    system_prompt = (
        f"You are 'Noir AI', Mudasir Shah's high-end personal concierge and AI Assistant.\n"
        f"Current Time: {current_time} (PKT - Pakistan Standard Time, UTC+5)\n"
        "Manage Mudasir's schedule and book meetings using the 'book_meeting_tool'.\n"
        "*** CRITICAL RULE FOR BOOKING MEETINGS ***:\n"
        "BEFORE you can ever call the 'book_meeting_tool', you ABSOLUTELY MUST have gathered ALL 3 of these details from the user:\n"
        "  1. Their Name\n"
        "  2. Their Email Address\n"
        "  3. The proposed time and date\n"
        "ESCAPE HATCH: If the user says 'no', 'nevermind', or clearly refuses to provide their email/details, DO NOT keep asking. Acknowledge their refusal politely and stop asking for details.\n"
        "CRITICAL SEQUENTIAL GATHERING: If multiple details are missing, DO NOT ask for all of them at once. "
        "Ask for them ONE BY ONE. For example, if you have nothing, just ask 'What is your name?'. Once they reply, ask 'What is your email?', and finally 'What time would you like to meet?'.\n"
        "ALWAYS check the conversation history first. Do not ask for information they have already provided.\n"
        "TIMEZONE CLARIFICATION: If the user suggests a time but doesn't specify a timezone, gently ask: 'Are you referring to [TIME] EST, or my local time (PKT)?'\n"
        "DATE CONFIRMATION RULE: Before calling the `book_meeting_tool`, ALWAYS repeat the final date and time back to the user to confirm (e.g., 'I have you down for Feb 26th at 2:00 PM PKT. Should I go ahead and lock this in?'). ONLY call the tool AFTER they explicitly agree.\n"
        "ERROR RECOVERY: If a tool returns an error (e.g., slot unavailable), do not panic or show raw JSON. Apologize gracefully, explain the time might be taken, and ask for an alternative time.\n"
        "IMPORTANT: When calling calendar tools, ALWAYS use the account name 'normal' and calendar ID 'primary'.\n"
        "TOOL USAGE WARNING: NEVER type raw tool calls like `<function=...>` or `(function=...` in your text.\n"
        "TONE & FORMAT INSTRUCTIONS:\n"
        "1. Speak confidently, professionally, and warmly. Think of yourself as a high-end concierge.\n"
        "2. BE CONCISE. Do not add fluff. Just state the schedule or ask for missing information cleanly.\n"
        "3. EMOJIS: Use emojis sparingly to be friendly (max 1 per message). Do not look like a spam bot."
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
    
    workflow.add_node("identify_intent", identify_intent)
    workflow.add_node("portfolio_chatbot", portfolio_chatbot)
    workflow.add_node("calendar_chatbot", calendar_chatbot)
    
    # Tool nodes - Combine all tools
    cal_tools = await get_calendar_tools()
    read_only_mcp_tools = [t for t in cal_tools if t.name in ["list-calendars", "list-events"]]
    all_tools = portfolio_tools + read_only_mcp_tools + [book_meeting_tool]
    workflow.add_node("tools", ToolNode(all_tools))
    
    workflow.add_edge(START, "identify_intent")
    workflow.add_conditional_edges("identify_intent", route_decision, ["portfolio_chatbot", "calendar_chatbot"])
    
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
