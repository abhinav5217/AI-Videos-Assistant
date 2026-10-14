# AI Video Assistant — OpenAI Only

Analyze a YouTube video or a local audio/video file. The app transcribes English or Hindi-English mixed (Hinglish) audio with OpenAI, generates a title and summary, extracts action items/decisions/open questions, and lets you ask questions about the transcript using RAG.

## Requirements

- Python 3.10–3.12 recommended.
- FFmpeg installed and available on `PATH` (Pydub uses it for media handling).
- One OpenAI API key with API access and billing configured. The same key is used for transcription, chat, and embeddings.
- YouTube downloads can fail due to platform restrictions, bot checks, age restrictions, or upstream `yt-dlp` changes. A local media file is an alternative.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your OpenAI API key as `OPENAI_API_KEY=...`. Do not commit `.env`.
4. Start the web app:

   ```bash
   streamlit run app.py
   ```

5. Or use the command line:

   ```bash
   python main.py
   ```

## Supported input

- A valid HTTP(S) video URL accepted by `yt-dlp`.
- A local audio/video file readable by FFmpeg, entered as a path on the machine running the app.
- Language options: `english` or `hinglish`. Hinglish mode asks OpenAI to preserve the Hindi-English language mix instead of translating it into English. Transcription output may still vary with audio clarity, accents, and code-switching.

## Notes

- OpenAI is used for speech transcription, chat completions, and embeddings; no Mistral, Sarvam, or local Whisper model is required.
- API usage may incur OpenAI charges. Check current model availability and pricing in your OpenAI account.
- Each transcript is indexed into its own ChromaDB collection, preventing retrieval across unrelated meetings. Reprocessing the same transcript replaces that transcript's chunks.
- The vector database and downloaded media are local runtime data and are excluded from Git.
- Live API calls and video downloads require your own key, network access, FFmpeg, and compatible installed dependencies.
