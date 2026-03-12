from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
import asyncio
import traceback

from orchestrator import orchestrate_query, get_graph
from voice import transcribe_audio, synthesize_speech

# Load environment variables
load_dotenv()

app = FastAPI(title="Noir AI Service")

# CORS — origins loaded exclusively from the ALLOWED_ORIGINS env var
_raw = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _raw.split(",") if o.strip()]
if not _allowed_origins:
    raise RuntimeError("ALLOWED_ORIGINS env var is not set. Set it to your frontend URL(s).")
print(f"[CORS] Allowed origins: {_allowed_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Pre-warm the LangGraph so the first request doesn't cold-start crash silently."""
    print("[startup] Pre-warming LangGraph...")
    try:
        await get_graph()
        print("[startup] LangGraph ready.")
    except Exception as e:
        print(f"[startup] ERROR: LangGraph failed to initialize: {e}")
        traceback.print_exc()
        # Don't crash the server — log it so Railway shows the real error

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.options("/chat")
async def chat_preflight(response: Response):
    return Response(status_code=200)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"[/chat] received message (session={request.session_id}): {request.message[:80]}")
    try:
        result = await asyncio.wait_for(
            orchestrate_query(request.message, session_id=request.session_id),
            timeout=55.0
        )
        print(f"[/chat] success, latency={result.get('trace', {}).get('total_latency_ms')}ms")
        return {
            "response": result["response"],
            "session_id": request.session_id,
            "trace": result["trace"]
        }
    except asyncio.TimeoutError:
        print("[/chat] TIMEOUT: orchestrate_query took >55s")
        return {
            "response": "The request timed out. Please try again.",
            "session_id": request.session_id,
            "trace": {"error": True, "detail": "timeout"}
        }
    except Exception as e:
        print(f"[/chat] CRITICAL ERROR: {str(e)}")
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
