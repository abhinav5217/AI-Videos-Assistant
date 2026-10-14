"""Meeting-scoped ChromaDB utilities for the AI Video Assistant."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = os.getenv("CHROMA_DIR", "vector_db")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_PREFIX = "meeting_"


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings lazily when a vector store is needed."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file or deployment environment.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.getenv("OPENAI_API_KEY"))


def _collection_name(meeting_id: str) -> str:
    """Return a stable, Chroma-compatible collection name for a meeting."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", meeting_id).strip("_")
    digest = hashlib.sha256(meeting_id.encode("utf-8")).hexdigest()[:10]
    return f"{COLLECTION_PREFIX}{safe_id[:40] or 'session'}_{digest}"


def build_vector_store(transcript: str, meeting_id: str | None = None) -> Chroma:
    """Create a dedicated persistent collection for this transcript.

    Rebuilding the same meeting_id replaces that meeting's old chunks, while
    other meetings remain isolated in their own collections.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build a vector store from an empty transcript.")

    meeting_id = meeting_id or hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:20]
    collection_name = _collection_name(meeting_id)
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True,
    )
    chunks = splitter.split_text(transcript.strip())
    if not chunks:
        raise ValueError("No text chunks were created from the transcript.")

    # Remove stale chunks only from this meeting's collection.
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    existing = store.get(include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        store.delete(ids=existing_ids)

    documents = [
        Document(
            page_content=chunk,
            metadata={"meeting_id": meeting_id, "chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]
    ids = [f"{meeting_id}_{index}" for index in range(len(documents))]
    store.add_documents(documents=documents, ids=ids)
    return store


def load_vector_store(meeting_id: str) -> Chroma:
    """Load a previously-created collection for the specified meeting."""
    if not meeting_id or not meeting_id.strip():
        raise ValueError("meeting_id is required to load a vector store.")
    return Chroma(
        collection_name=_collection_name(meeting_id),
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    """Build a similarity retriever with a validated result count."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
