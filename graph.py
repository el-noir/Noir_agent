from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from agents.router import router_agent, route_query
from agents.portfolio import portfolio_agent
from agents.project import project_agent

# Define the state object for the graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    metadata: dict

def create_portfolio_graph():
    """
    Constructs the LangGraph for the portfolio AI.
    """
    # 1. Initialize StateGraph
    workflow = StateGraph(AgentState)

    # 2. Add Nodes (Agents/Functions)
    workflow.add_node("router", router_agent)
    workflow.add_node("portfolio", portfolio_agent)
    workflow.add_node("project", project_agent)

    # 3. Add Edges
    workflow.add_edge(START, "router")

    # Conditional routing based on 'intent'
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "ABOUT": "portfolio",
            "PROJECT": "project",
            "OUT_OF_SCOPE": END, # Or a dedicated handler later
            "DEFAULT": "portfolio"
        }
    )

    # After agents finish, end for now (later: Memory update, Self-eval)
    workflow.add_edge("portfolio", END)
    workflow.add_edge("project", END)

    # Compile the graph
    app = workflow.compile()
    
    return app
