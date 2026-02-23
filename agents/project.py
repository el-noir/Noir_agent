from langchain_core.messages import HumanMessage, AIMessage

def project_agent(state: dict):
    """
    Handles deep dives into specific projects or technical architecture.
    Will use RAG against project READMEs, architecture docs, etc.
    """
    print("---PROJECT DEEP-DIVE AGENT---")
    messages = state["messages"]
    
    # Placeholder for RAG retrieval
    # In Phase 1, we simulate retrieving technical docs
    
    retrieved_docs = """
    Document: Network Scanner
    Architecture: Uses raw sockets in C for high performance. Multi-threaded design to scan subnets concurrently.
    Trade-offs: Selected C over Python for raw socket access speed, at the cost of memory safety complexity.
    """
    
    # Fake response generation based on the mock docs
    last_msg = messages[-1].content.lower()
    
    if "scanner" in last_msg or "network" in last_msg:
         response_text = "The network scanner is built in C using raw sockets and a multi-threaded design for concurrent subnet scanning. The main trade-off was choosing C for performance over Python, which introduced memory management complexity but significantly reduced scan times."
    else:
         response_text = "I've built several complex projects ranging from low-level network scanners to high-level multi-agent AI systems. Which project would you like to deep-dive into?"
         
    response_msg = AIMessage(content=response_text, name="project_agent")
    
    return {"messages": [response_msg]}
