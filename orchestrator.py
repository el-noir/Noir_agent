import time
import json
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

from tools.implementation import (
    getProfile,
    listProjects,
    explainProject,
    getAvailability,
    analyzeJobFit
)

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

# Convert Python functions to LangChain Tools
@tool
def get_profile_tool() -> str:
    """Retrieves Mudasir's basic profile information, bio, and general experience."""
    return json.dumps(getProfile())

@tool
def list_projects_tool(filters: list[str] = None) -> str:
    """Searches the portfolio for projects matching optional filters like ['React', 'Python', 'AI']."""
    return json.dumps(listProjects(filters))

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

tools = [
    get_profile_tool,
    list_projects_tool,
    explain_project_tool,
    get_availability_tool,
    analyze_job_fit_tool
]

llm_with_tools = llm.bind_tools(tools)

def orchestrate_query(user_message: str):
    """
    Core orchestrator:
    1. Sends user query to LLM to select a tool.
    2. Executes tool if selected.
    3. Synthesizes a response with the trace metadata.
    """
    
    start_time = time.time()
    
    if not os.getenv("GROQ_API_KEY"):
         return {
             "response": "Error: GROQ_API_KEY is not set in the environment. Please add it to your .env file.",
             "trace": {
                 "intent_detected": "Error",
                 "tool_selected": "None",
                 "tool_args": {},
                 "tool_latency_ms": 0,
                 "total_latency_ms": round((time.time() - start_time) * 1000, 2)
             }
         }
    
    # 1. Ask LLM to act on the query
    try:
        initial_response = llm_with_tools.invoke([
            ("system", "You are the AI interface for Mudasir Shah's technical portfolio. Determine if a tool is needed to answer the user's question. If so, call it."),
            ("human", user_message)
        ])
    except Exception as e:
        return {
            "response": f"Failed to connect to LLM: {str(e)}",
            "trace": {
                "intent_detected": "Error",
                "tool_selected": "None",
                "tool_args": {},
                "tool_latency_ms": 0,
                "total_latency_ms": round((time.time() - start_time) * 1000, 2)
            }
        }
    
    trace = {
        "intent_detected": "Direct Answer",
        "tool_selected": "None",
        "tool_args": {},
        "tool_latency_ms": 0,
        "total_latency_ms": 0
    }
    
    final_answer = ""
    
    # Check if a tool was called
    if hasattr(initial_response, 'tool_calls') and initial_response.tool_calls:
        tool_call = initial_response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        trace["intent_detected"] = "Tool Execution"
        trace["tool_selected"] = tool_name
        trace["tool_args"] = tool_args
        
        tool_start = time.time()
        
        # 2. Execute the selected tool
        tool_result_json_str = ""
        try:
            # Map name back to the actual function
            target_tool = next((t for t in tools if t.name == tool_name), None)
            if target_tool:
                tool_result_json_str = target_tool.invoke(tool_args)
            else:
                 tool_result_json_str = json.dumps({"error": f"Tool {tool_name} not found."})
        except Exception as e:
             tool_result_json_str = json.dumps({"error": str(e)})
             
        tool_end = time.time()
        trace["tool_latency_ms"] = round((tool_end - tool_start) * 1000, 2)
        
        # 3. Synthesize final response based on tool output
        try:
            synthesis_response = llm.invoke([
                ("system", "You are the AI interface for Mudasir Shah's technical portfolio. Use the provided JSON tool output to give a natural, professional answer to the user. Do not leak raw JSON unless specifically formatting it nicely."),
                ("human", f"User question: {user_message}\n\nTool Output: {tool_result_json_str}")
            ])
            
            final_answer = synthesis_response.content
        except Exception as e:
             final_answer = f"I retrieved the information but failed to synthesize it: {str(e)}. Raw Result: {tool_result_json_str}"
        
    else:
        # No tool called, just synthesized
        final_answer = initial_response.content
        
    end_time = time.time()
    trace["total_latency_ms"] = round((end_time - start_time) * 1000, 2)
    
    return {
        "response": final_answer,
        "trace": trace
    }
