import time
from graph import create_portfolio_graph
from langchain_core.messages import HumanMessage
import os

# Lazy graph initialization to avoid import-time loop conflicts
_graph_app = None

async def get_graph():
    global _graph_app
    if _graph_app is None:
        _graph_app = await create_portfolio_graph()
    return _graph_app

async def orchestrate_query(user_message: str, session_id: str = "default_session"):
    """
    Core orchestrator using LangGraph multi-agent system (async) with memory.
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
    
    try:
        # Prepare inputs for the graph
        inputs = {"messages": [HumanMessage(content=user_message)], "trace": {}}
        config = {"configurable": {"thread_id": session_id}}
        
        # Invoke the graph (async)
        app = await get_graph()
        result = await app.ainvoke(inputs, config=config)
        
        final_answer = result["messages"][-1].content
        trace = result.get("trace", {
            "intent_detected": "Unknown",
            "tool_selected": "None",
            "tool_args": {},
            "active_agent": "Unknown"
        })
        
        # Add latency
        end_time = time.time()
        trace["total_latency_ms"] = round((end_time - start_time) * 1000, 2)
        
        return {
            "response": final_answer,
            "trace": trace
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_str = str(e)
        if "rate_limit_exceeded" in err_str or "429" in err_str:
            friendly = "I'm handling a lot of requests right now — please try again in a moment."
        else:
            friendly = f"System Error: {err_str}"
        return {
            "response": friendly,
            "trace": {
                "intent_detected": "Error",
                "tool_selected": "None",
                "tool_args": {},
                "total_latency_ms": round((time.time() - start_time) * 1000, 2)
             }
        }

