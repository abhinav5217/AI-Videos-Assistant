"""OpenAI speech-to-text for English and Hindi-English mixed (Hinglish) audio."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env before reading configuration.
load_dotenv()

TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Create the OpenAI client lazily, with a clear configuration error."""
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add your OpenAI API key to the .env file "
            "or set it in your deployment environment."
        )
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Transcribe one audio chunk with OpenAI, retaining Hinglish code-switching."""
    normalized_language = (language or "").strip().lower()
    if normalized_language not in {"english", "hinglish"}:
        raise ValueError("Language must be 'english' or 'hinglish'.")
    if not os.path.isfile(chunk_path):
        raise FileNotFoundError(f"Audio chunk not found: {chunk_path}")

    if normalized_language == "hinglish":
        prompt = (
            "The speaker may switch between Hindi and English (Hinglish). "
            "Transcribe faithfully and keep the original language mix. Do not translate. "
            "Write Hindi words in Devanagari where appropriate and English words in Latin script. "
            "Preserve names, technical terms, and meaning."
        )
    else:
        prompt = "Transcribe the spoken English faithfully. Preserve names and technical terms."

    client = get_openai_client()
    try:
        with open(chunk_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                prompt=prompt,
            )
    except Exception as exc:
        # Do not log API keys or request headers.
        raise RuntimeError(f"OpenAI transcription failed for {Path(chunk_path).name}: {exc}") from exc

    text = getattr(response, "text", "") or ""
    return text.strip()


def transcribe_all(chunks: list[str], language: str = "english") -> str:
    """Transcribe all audio chunks and combine them in their original order."""
    if not chunks:
        raise ValueError("No audio chunks were provided for transcription.")
    normalized_language = (language or "").strip().lower()
    if normalized_language not in {"english", "hinglish"}:
        raise ValueError("Language must be 'english' or 'hinglish'.")

    print(f"Using OpenAI ({TRANSCRIPTION_MODEL}) for {normalized_language} transcription.")
    transcript_parts: list[str] = []
    for index, chunk_path in enumerate(chunks, start=1):
        print(f"Transcribing chunk {index}/{len(chunks)}...")
        text = transcribe_chunk(chunk_path, language=normalized_language)
        if text:
            transcript_parts.append(text)

    transcript = "\n\n".join(transcript_parts).strip()
    if not transcript:
        raise RuntimeError("OpenAI transcription returned no text. Check the audio and selected language.")
    print("Transcription complete.")
    return transcript
