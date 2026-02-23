from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

from graph import create_portfolio_graph

# Load environment variables
load_dotenv()

app = FastAPI(title="Portfolio AI Service")

# Configure CORS for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LangGraph
app_graph = create_portfolio_graph()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Minimal synchronous execution for now, stream later
    inputs = {"messages": [("user", request.message)]}
    
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Run the graph
    result = app_graph.invoke(inputs, config=config)
    
    # Extract the last AI message
    last_msg = result["messages"][-1]
    
    return {
        "response": last_msg.content,
        "session_id": request.session_id,
        "metadata": result.get("metadata", {})
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
