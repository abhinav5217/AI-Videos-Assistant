"""Command-line entry point and reusable pipeline for AI Video Assistant."""
from __future__ import annotations

from dotenv import load_dotenv

# Load configuration before importing modules that read environment variables.
load_dotenv()

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import cleanup_chunks, process_input


def run_pipeline(source: str, language: str = "english") -> dict:
    """Run the full analysis pipeline and return the results plus a RAG chain."""
    if not source or not source.strip():
        raise ValueError("Please provide a video URL or local file path.")
    language = (language or "english").strip().lower()
    if language not in {"english", "hinglish"}:
        raise ValueError("Language must be 'english' or 'hinglish'.")

    chunks = []
    try:
        print("Starting AI Video Assistant...")
        chunks = process_input(source.strip())
        transcript = transcribe_all(chunks, language=language)
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        rag_chain = build_rag_chain(transcript)
        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }
    finally:
        cleanup_chunks(chunks)


def main() -> None:
    source = input("Enter video URL or local file path: ").strip()
    language = input("Language (english/hinglish) [english]: ").strip().lower() or "english"
    try:
        result = run_pipeline(source, language)
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        return

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}\n\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}\n" + "=" * 60)
    print("Chat with your meeting (type 'exit' to quit).")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            print(f"Assistant: {ask_question(result['rag_chain'], question)}\n")
        except Exception as exc:
            print(f"Could not answer that question: {exc}")


if __name__ == "__main__":
    main()
