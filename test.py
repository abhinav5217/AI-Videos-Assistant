"""Manual smoke-test entry point. No external URL is hard-coded."""
from dotenv import load_dotenv

load_dotenv()

from main import run_pipeline
from core.rag_engine import ask_question


def main() -> None:
    source = input("Enter video URL or local file path: ").strip()
    language = input("Language (english/hinglish) [english]: ").strip().lower() or "english"
    try:
        result = run_pipeline(source, language)
    except Exception as exc:
        print(f"Test run failed: {exc}")
        return

    print("\n" + "=" * 60)
    for label, key in [
        ("TITLE", "title"),
        ("SUMMARY", "summary"),
        ("TRANSCRIPT", "transcript"),
        ("ACTION ITEMS", "action_items"),
        ("KEY DECISIONS", "key_decisions"),
        ("OPEN QUESTIONS", "open_questions"),
    ]:
        print(f"\n{label}\n{'-' * 60}\n{result[key]}")

    while True:
        question = input("\nAsk about the meeting (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            try:
                print(ask_question(result["rag_chain"], question))
            except Exception as exc:
                print(f"Question failed: {exc}")


if __name__ == "__main__":
    main()
