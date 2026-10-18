"""Download/extract audio, normalize it, and split it into transcription chunks."""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "downloads")
CHUNK_MINUTES = int(os.getenv("AUDIO_CHUNK_MINUTES", "10"))
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_youtube_audio(url: str) -> str:
    """Download a video's audio as WAV and return the verified final path."""
    if not _is_url(url):
        raise ValueError("Please provide a valid http(s) video URL.")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = str(info.get("id", ""))
            expected = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")
            if os.path.isfile(expected):
                return expected
            # yt-dlp versions may report the final path differently; verify candidates.
            candidates = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}.*"))
            wav_candidates = [path for path in candidates if path.lower().endswith(".wav") and os.path.isfile(path)]
            if wav_candidates:
                return max(wav_candidates, key=os.path.getmtime)
            raise FileNotFoundError("yt-dlp completed but the converted WAV file could not be found. Check FFmpeg installation and download logs.")
    except Exception as exc:
        raise RuntimeError(f"Could not download/extract audio from the video URL: {exc}") from exc


def convert_to_wav(input_path: str) -> str:
    """Convert a local supported audio/video file into mono 16 kHz WAV."""
    if not input_path or not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    try:
        audio = AudioSegment.from_file(input_path)
        if len(audio) == 0:
            raise ValueError("The input media contains no audio.")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not read or convert the input media. Check that FFmpeg is installed. Details: {exc}") from exc
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Audio conversion did not produce a valid WAV file.")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = CHUNK_MINUTES) -> list[str]:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if chunk_minutes < 1:
        raise ValueError("chunk_minutes must be at least 1.")
    audio = AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(16000)
    if len(audio) == 0:
        raise ValueError("The input audio is empty.")

    chunk_dir = tempfile.mkdtemp(prefix="video_assistant_chunks_")
    chunk_ms = chunk_minutes * 60 * 1000
    paths: list[str] = []
    try:
        for index, start in enumerate(range(0, len(audio), chunk_ms)):
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:04d}.wav")
            audio[start:start + chunk_ms].export(chunk_path, format="wav")
            paths.append(chunk_path)
        return paths
    except Exception:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise


def cleanup_chunks(chunks: list[str] | None) -> None:
    """Remove temporary chunk files and their containing temporary directory."""
    if not chunks:
        return
    directories = {os.path.dirname(os.path.abspath(path)) for path in chunks}
    for path in chunks:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    for directory in directories:
        if os.path.basename(directory).startswith("video_assistant_chunks_"):
            shutil.rmtree(directory, ignore_errors=True)


def process_input(source: str) -> list[str]:
    """Prepare either a video URL or a local media file and return chunk paths."""
    if not source or not source.strip():
        raise ValueError("Please provide a YouTube URL or local file path.")
    source = source.strip()
    intermediate_path = None
    downloaded = False
    if _is_url(source):
        print("Detected video URL. Downloading audio...")
        intermediate_path = download_youtube_audio(source)
        downloaded = True
    else:
        print("Detected local file. Converting to WAV...")
        intermediate_path = convert_to_wav(source)

    try:
        print("Preparing audio chunks...")
        chunks = chunk_audio(intermediate_path)
        print(f"Audio ready — {len(chunks)} chunk(s) created.")
        return chunks
    finally:
        # Intermediate files are no longer needed once chunk files have been written.
        if intermediate_path and os.path.isfile(intermediate_path):
            try:
                os.remove(intermediate_path)
            except OSError:
                pass



def process_uploaded_file(uploaded_file) -> list[str]:
    """Save an uploaded media file temporarily and create audio chunks."""
    if uploaded_file is None:
        raise ValueError("Please upload an audio or video file.")

    # Allow common audio and video formats supported by FFmpeg.
    allowed_extensions = {
        ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
        ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"
    }

    # Get the original file extension.
    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in allowed_extensions:
        raise ValueError("Unsupported file format. Please upload a supported audio or video file.")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Store the uploaded file temporarily on the Render server.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            dir=DOWNLOAD_DIR,
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getvalue())

        # Reuse your existing conversion and chunking pipeline.
        return process_input(temp_path)

    finally:
        # Remove the original uploaded file after processing.
        if temp_path and os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
