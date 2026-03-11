"""
Voice I/O helpers for Noir AI.

STT: Groq Whisper (whisper-large-v3-turbo)
TTS: ElevenLabs HTTP API  (returns base64-encoded MP3, or None if not configured)
"""

import os
import base64
import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


async def transcribe_audio(audio_bytes: bytes, filename: str = "recording.webm") -> str:
    """Transcribe audio bytes using Groq Whisper and return the transcript text."""
    transcription = await _groq.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="text",
    )
    return transcription.strip()


async def synthesize_speech(text: str) -> str | None:
    """
    Convert text to speech via ElevenLabs.
    Returns base64-encoded MP3 bytes, or None when ElevenLabs is not configured.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRTpX")  # Default: George

    if not api_key or api_key.startswith("your_"):
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode()
        return None
