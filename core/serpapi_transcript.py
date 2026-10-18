
"""Fetch a YouTube transcript using SerpApi."""
import os
import re
from urllib.parse import parse_qs, urlparse

import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def extract_youtube_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from a URL."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = (
                parts[1]
                if len(parts) >= 2
                and parts[0] in {"shorts", "embed", "live"}
                else ""
            )
    else:
        raise ValueError("Please enter a valid YouTube URL.")

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id or ""):
        raise ValueError("Could not extract the YouTube video ID.")

    return video_id


def get_youtube_transcript(url: str, language: str = "english") -> str:
    """Fetch transcript text from SerpApi."""
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not configured.")

    video_id = extract_youtube_video_id(url)

    # Use the requested transcript language.
    language_code = "hi" if language.lower() == "hinglish" else "en"

    try:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params={
                "engine": "youtube_video_transcript",
                "v": video_id,
                "language_code": language_code,
                "api_key": api_key,
            },
            timeout=35,
        )
        response.raise_for_status()
        data = response.json()

    except requests.Timeout as exc:
        raise RuntimeError("SerpApi request timed out. Please retry.") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 401:
            message = "SerpApi API key was rejected. Check your environment variable."
        elif status == 429:
            message = "SerpApi rate limit or quota may have been reached."
        else:
            message = f"SerpApi returned HTTP status {status or 'unknown'}."
        raise RuntimeError(message) from exc
    except requests.RequestException as exc:
        raise RuntimeError("Could not connect to SerpApi.") from exc
    except ValueError as exc:
        raise RuntimeError("SerpApi returned an invalid JSON response.") from exc

    if data.get("error"):
        # Never print the API key or request URL in an error message.
        raise RuntimeError(f"SerpApi error: {str(data['error'])[:250]}")

    segments = data.get("transcript", [])
    if not isinstance(segments, list):
        segments = []

    # Join transcript snippets into one string for the existing AI pipeline.
    parts = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") or item.get("text") or item.get("content")
        if isinstance(snippet, str) and snippet.strip():
            parts.append(snippet.strip())

    transcript = " ".join(parts).strip()

    if not transcript:
        raise RuntimeError(
            "No transcript was returned. The video may not have an "
            "accessible transcript in the selected language."
        )

    return transcript
