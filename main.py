from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

from orchestrator import orchestrate_query
from voice import transcribe_audio, synthesize_speech

# Load environment variables
load_dotenv()

app = FastAPI(title="Noir AI Service")

# CORS — allow the deployed frontend origin + localhost for dev
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
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

@app.post("/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """Accept an audio file, transcribe it, run through the AI graph, and return TTS audio."""
    audio_bytes = await audio.read()
    filename = audio.filename or "recording.webm"

    transcript = await transcribe_audio(audio_bytes, filename=filename)

    result = await orchestrate_query(transcript, session_id=session_id)

    audio_b64 = await synthesize_speech(result["response"])

    return {
        "transcript": transcript,
        "response": result["response"],
        "audio_base64": audio_b64,
        "session_id": session_id,
        "trace": result["trace"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
