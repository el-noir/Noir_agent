from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

from orchestrator import orchestrate_query

# Load environment variables
load_dotenv()

app = FastAPI(title="Portfolio AI Service")

# Configure CORS - loosened for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Pass user message and session to orchestrator
        result = await orchestrate_query(request.message, session_id=request.session_id)
        
        return {
            "response": result["response"],
            "session_id": request.session_id,
            "trace": result["trace"]
        }
    except Exception as e:
        print(f"CRITICAL ERROR IN /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"Internal Server Error: {str(e)}",
            "session_id": request.session_id,
            "trace": {"error": True, "detail": str(e)}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
