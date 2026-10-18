"""AI Video Assistant — meeting analysis with multi-chat session history."""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from core.serpapi_transcript import get_youtube_transcript
import streamlit as st
from dotenv import load_dotenv

# Load environment variables before importing modules that read them.
load_dotenv()

from utils.audio_processor import cleanup_chunks, process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Production-inspired styling ───────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #0b0c10;
  --surface: #12141b;
  --surface-2: #191c25;
  --surface-3: #202431;
  --border: #2b3040;
  --text: #f3f5fb;
  --muted: #9ba4b8;
  --accent: #9b7bff;
  --accent-2: #42d6d0;
  --success: #43d39e;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 75% 0%, rgba(115, 83, 220, .13), transparent 36%),
    var(--bg);
}
[data-testid="stSidebar"] {
  background: #101117;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.025em; }
p, li, label, span { line-height: 1.55; }
.hero-kicker {
  color: var(--accent-2); font-size: .72rem; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase;
}
.hero-title {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.05;
  letter-spacing: -.055em; margin: .35rem 0 .65rem;
  background: linear-gradient(100deg, #fff 10%, #c7b8ff 55%, #67e3df 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: var(--muted); font-size: 1rem; max-width: 680px; }
.section-title {
  font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem;
  font-weight: 700; margin-bottom: .65rem;
}
.eyebrow {
  color: var(--muted); font-size: .7rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
}
.brand {
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: 1.65rem; line-height: 1.05; letter-spacing: -.04em;
}
.brand span { color: var(--accent); }
.sidebar-caption { color: var(--muted); font-size: .78rem; }
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--border) !important;
  background: rgba(18,20,27,.82);
  border-radius: 16px !important;
}
.stButton > button {
  border-radius: 10px; min-height: 2.6rem; font-weight: 600;
  border: 1px solid var(--border); transition: all .15s ease;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(110deg, #8d6bff, #6d56d9);
  border: 0; color: white;
}
.stButton > button:hover { border-color: var(--accent); transform: translateY(-1px); }
.stTextInput input, .stSelectbox [data-baseweb="select"],
.stTextArea textarea, [data-testid="stFileUploader"] section {
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stChatMessage"] {
  border: 1px solid var(--border); border-radius: 14px;
  padding: .8rem 1rem; margin-bottom: .7rem;
}
[data-testid="stChatInput"] {
  border-color: var(--border) !important;
  background: var(--surface-2) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
hr { border-color: var(--border) !important; }
div[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border);
  padding: 1rem; border-radius: 14px;
}
.small-muted { color: var(--muted); font-size: .82rem; }
.status-pill {
  display: inline-block; padding: .28rem .6rem; border-radius: 999px;
  border: 1px solid rgba(67,211,158,.25); background: rgba(67,211,158,.08);
  color: #76e6b8; font-size: .72rem; font-weight: 700;
}
[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ──────────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = {}
if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def short_title(value: str, limit: int = 29) -> str:
    value = " ".join(str(value or "New meeting").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_step_bar(label: str, key: str, icon: str) -> None:
    state = st.session_state.pipeline_steps.get(key, "pending")
    color = {"done": "#43d39e", "active": "#9b7bff", "error": "#ff6b7a"}.get(state, "#555c6d")
    state_text = {"done": "Complete", "active": "In progress", "error": "Error"}.get(state, "Waiting")
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:.65rem;padding:.55rem .7rem;
        margin:.25rem 0;border:1px solid #2b3040;border-radius:9px;background:#191c25">
        <span style="color:{color};font-size:.8rem">●</span>
        <span style="flex:1;font-size:.82rem">{icon} {label}</span>
        <span style="color:#9ba4b8;font-size:.68rem">{state_text}</span></div>""",
        unsafe_allow_html=True,
    )


def current_conversation():
    chat_id = st.session_state.active_chat_id
    if chat_id and chat_id in st.session_state.conversations:
        return st.session_state.conversations[chat_id]
    return None


def format_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%b %d · %I:%M %p")
    except (ValueError, TypeError):
        return ""


# ── Sidebar: brand, multi-chat history, and new input ──────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">◩ AI <span>Video</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">MEETING INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("＋  New meeting chat", type="primary", use_container_width=True, key="new_chat_btn"):
        st.session_state.active_chat_id = None
        st.session_state.last_error = None
        st.rerun()

    st.markdown('<div class="eyebrow" style="margin:.9rem 0 .45rem">Your chats</div>', unsafe_allow_html=True)
    conversations = st.session_state.conversations
    if conversations:
        for chat_id, conversation in reversed(list(conversations.items())):
            is_active = chat_id == st.session_state.active_chat_id
            label = ("● " if is_active else "○ ") + short_title(conversation.get("title", "Meeting"))
            if st.button(
                label,
                key=f"open_chat_{chat_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                help=conversation.get("title", "Meeting"),
            ):
                st.session_state.active_chat_id = chat_id
                st.session_state.last_error = None
                st.rerun()
            timestamp = format_timestamp(conversation.get("created_at"))
            if timestamp:
                st.markdown(
                    f'<div style="margin:-.45rem 0 .45rem 1.5rem;color:#727b90;font-size:.68rem">{timestamp}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="sidebar-caption">Your analysed meetings will appear here. Start a new meeting to create your first chat.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="eyebrow" style="margin-bottom:.55rem">Add a meeting</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["YouTube URL", "Upload audio/video"],
        key="input_mode",
        label_visibility="collapsed",
    )
    source = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL",
            placeholder="Paste a video URL",
            key="youtube_url_input",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose media",
            type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4", "mov", "mkv", "webm", "mpeg", "mpg"],
            key="media_upload_input",
            help="Supported audio/video formats. Large files need more processing time.",
            label_visibility="collapsed",
        )

    language = st.selectbox("Transcript language", ["english", "hinglish"], key="transcript_language")
    run_btn = st.button("Analyse meeting  →", type="primary", use_container_width=True, key="analyse_meeting_btn")

    if st.session_state.pipeline_steps:
        st.markdown("---")
        st.markdown('<div class="eyebrow" style="margin-bottom:.4rem">Pipeline status</div>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio", "◉", "Audio processing"),
            ("transcript", "▤", "Transcription"),
            ("title", "✦", "Title generation"),
            ("summary", "▣", "Summary"),
            ("extract", "⌕", "Meeting insights"),
            ("rag", "◇", "Search & chat"),
        ]:
            render_step_bar(label, step, icon)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-kicker">Your workspace · Meeting intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Turn conversations into clarity.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Transcribe recordings, capture decisions, discover follow-ups, and ask questions about every meeting — all in one workspace.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Analysis pipeline: every successful analysis creates a separate chat ────────
if run_btn:
    if input_mode == "YouTube URL" and not source.strip():
        st.error("Paste a YouTube URL first.")
    elif input_mode == "Upload audio/video" and uploaded_file is None:
        st.error("Choose an audio or video file first.")
    else:
        st.session_state.pipeline_steps = {}
        st.session_state.last_error = None
        chunks = []
        temp_upload_path = None
        pipeline_placeholder = st.empty()

        def update_step(key: str, state: str) -> None:
            st.session_state.pipeline_steps[key] = state

        
        try:
            with st.status(
                "Analysing your meeting…",
                expanded=True,
            ) as pipeline_status:

                transcript = ""

                # Step 1: Handle uploaded audio/video files as before.
                if input_mode == "Upload audio/video":
                    update_step("audio", "active")
                    pipeline_placeholder.info(
                        "Preparing the uploaded media…"
                    )

                    extension = Path(uploaded_file.name).suffix.lower()
                    if not extension:
                        raise ValueError(
                            "The uploaded file must have a file extension."
                        )

                    with tempfile.NamedTemporaryFile(
                        suffix=extension,
                        prefix="uploaded_media_",
                        delete=False,
                    ) as temp_file:
                        temp_upload_path = temp_file.name
                        temp_file.write(uploaded_file.getbuffer())

                    chunks = process_input(temp_upload_path)
                    update_step("audio", "done")

                else:
                    # Step 2: Try SerpApi before downloading YouTube audio.
                    update_step("audio", "active")
                    update_step("transcript", "active")

                    pipeline_placeholder.info(
                        "Trying to fetch the YouTube transcript…"
                    )

                    try:
                        transcript = get_youtube_transcript(
                            source.strip(),
                            language=language,
                        )

                        update_step("audio", "done")
                        update_step("transcript", "done")

                        pipeline_placeholder.success(
                            "Transcript fetched successfully."
                        )

                    except Exception as serpapi_error:
                        # Step 3: Fall back to the existing audio pipeline.
                        update_step("transcript", "error")
                        update_step("audio", "active")

                        pipeline_placeholder.warning(
                            "Transcript API failed. Trying audio processing…"
                        )

                        try:
                            chunks = process_input(source.strip())
                            update_step("audio", "done")

                        except Exception as audio_error:
                            raise RuntimeError(
                                "Transcript retrieval and audio processing "
                                "both failed. "
                                f"Transcript error: {serpapi_error}. "
                                f"Audio error: {audio_error}"
                            ) from audio_error

                # Step 4: Transcribe audio only if no transcript was returned.
                if not transcript.strip():
                    update_step("transcript", "active")
                    pipeline_placeholder.info("Transcribing audio…")

                    transcript = transcribe_all(chunks, language)
                    update_step("transcript", "done")

                if not transcript or not transcript.strip():
                    raise ValueError(
                        "No transcript was generated. Check the media, "
                        "API quota, and selected language."
                    )

                # Step 5: Keep the existing AI analysis pipeline.
                update_step("title", "active")
                title = generate_title(transcript)
                update_step("title", "done")

                update_step("summary", "active")
                summary_text = summarize(transcript)
                update_step("summary", "done")

                update_step("extract", "active")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)
                update_step("extract", "done")

                update_step("rag", "active")
                rag_chain = build_rag_chain(transcript)
                update_step("rag", "done")

                pipeline_status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )


            # Store the whole meeting under its own chat ID; creating a new chat
            # never overwrites the existing conversations dictionary.
            chat_id = make_chat_id()
            st.session_state.conversations[chat_id] = {
                "title": str(title or "Untitled meeting"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "result": {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary_text,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                },
                "messages": [],
            }
            st.session_state.active_chat_id = chat_id
            pipeline_placeholder.success("Meeting analysed. A new chat has been created; your previous chats are still in the sidebar.")
            time.sleep(0.25)
            st.rerun()

        except Exception as exc:
            st.session_state.last_error = str(exc)
            for step in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(step) == "active":
                    st.session_state.pipeline_steps[step] = "error"
            pipeline_placeholder.error(f"Analysis failed: {exc}")
        finally:
            cleanup_chunks(chunks)
            if temp_upload_path and os.path.isfile(temp_upload_path):
                try:
                    os.remove(temp_upload_path)
                except OSError:
                    pass


# ── Main workspace: active meeting ────────────────────────────────────────────
conversation = current_conversation()

if conversation:
    result = conversation["result"]
    st.markdown('<div class="eyebrow">Active meeting</div>', unsafe_allow_html=True)
    st.subheader(result["title"])
    st.markdown('<span class="status-pill">✓ Analysis complete</span>', unsafe_allow_html=True)
    st.markdown("")

    # Summary and meeting metadata
    summary_col, transcript_col = st.columns([1.35, 1], gap="large")
    with summary_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">✦ Meeting summary</div>', unsafe_allow_html=True)
            st.markdown(result["summary"])
    with transcript_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">▤ Transcript</div>', unsafe_allow_html=True)
            with st.expander("Read full transcript"):
                st.text(result["transcript"])

    action_col, decision_col, questions_col = st.columns(3, gap="medium")
    with action_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">✓ Action items</div>', unsafe_allow_html=True)
            st.markdown(result["action_items"])
    with decision_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">◇ Key decisions</div>', unsafe_allow_html=True)
            st.markdown(result["key_decisions"])
    with questions_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">? Open questions</div>', unsafe_allow_html=True)
            st.markdown(result["open_questions"])

    st.markdown("---")
    st.markdown('<div class="section-title">💬 Chat with this meeting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-muted">Ask follow-up questions about this transcript. Start a new meeting chat from the sidebar to keep another meeting separate.</div>',
        unsafe_allow_html=True,
    )

    for message in conversation["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this meeting…", key=f"chat_input_{st.session_state.active_chat_id}")
    if question and question.strip():
        conversation["messages"].append({"role": "user", "content": question.strip()})
        with st.spinner("Thinking through your meeting…"):
            try:
                answer = ask_question(result["rag_chain"], question.strip())
            except Exception as exc:
                answer = f"I couldn't answer that right now. Please try again. Details: {exc}"
        conversation["messages"].append({"role": "assistant", "content": answer})
        st.rerun()

else:
    # Helpful empty state for the first visit or a newly created chat.
    with st.container(border=True):
        st.markdown('<div class="hero-kicker">Welcome to your workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title" style="font-size:2.2rem">Your meetings, organised.</div>', unsafe_allow_html=True)
        st.markdown(
            "Add a meeting from the sidebar to get a transcript, a concise summary, action items, key decisions, open questions, and a dedicated Q&A chat."
        )
        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🎙️ Transcribe")
            st.caption("Turn audio or video into readable text.")
        with c2:
            st.markdown("### ✦ Understand")
            st.caption("Extract decisions, tasks, and follow-ups.")
        with c3:
            st.markdown("### 💬 Ask")
            st.caption("Ask questions in a chat dedicated to each meeting.")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

st.markdown(
    '<div style="text-align:center;color:#727b90;font-size:.72rem;padding:2rem 0 1rem">AI Video Assistant · Meeting intelligence workspace</div>',
    unsafe_allow_html=True,
)
