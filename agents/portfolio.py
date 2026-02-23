from langchain_core.messages import SystemMessage, HumanMessage

def portfolio_agent(state: dict):
    """
    Handles questions about Bio, Skills, Experience, and Availability.
    Uses basic context or RAG (to be connected).
    """
    print("---PORTFOLIO AGENT---")
    messages = state["messages"]
    
    # In Phase 1, we use static context or a simple RAG call
    system_prompt = SystemMessage(
        content="""You are the Portfolio Agent for a top-tier software engineer.
        Your goal is to answer questions about their background, skills, and experience confidently and accurately.
        
        Answer based on the following context (assume it was retrieved via RAG):
        - Current Focus: AI Engineering, Multi-Agent Architectures, Full-stack development.
        - Tech Stack: React, Next.js, Node.js, Python, LangGraph, FastAPI.
        - Experience: Built complex AI products and scalable systems.
        - Professional Strengths: System design, fast iteration, clean architecture.
        
        Keep answers professional, insightful, and concise. Highlight engineering depth.
        """
    )
    
    # We don't invoke the LLM yet to keep it simple and testable without keys
    # Fake a response for now based on intent
    intent = state.get("intent", "ABOUT")
    
    if intent == "AVAILABILITY":
        response_text = "I am currently open to discussing new opportunities, particularly roles involving complex AI architectures or full-stack engineering. What kind of role are you hiring for?"
    elif intent == "COLLABORATION":
        response_text = "I love collaborating with smart teams! I typically bring strong system design skills and a focus on shipping fast. Let's talk about the project."
    else:
        response_text = "I'm a software engineer specializing in AI-native applications and robust full-stack systems using Next.js, FastAPI, and LangGraph. I focus on engineering rigor—building scalable, maintainable architectures."
        
    response_msg = HumanMessage(content=response_text, name="portfolio_agent") # Using HumanMessage for testing if no LLM
    
    # Append the response to the state
    return {"messages": [response_msg]}
