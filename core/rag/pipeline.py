import hashlib

from core.rag.chunker import chunk_transcript
from core.rag.vector_store import index_chunks
from core.rag.retriever import retrieve_chunks
from core.rag.generator import generate_answer


def create_meeting_id(
    transcript: str
) -> str:

    return hashlib.sha256(
        transcript.encode("utf-8")
    ).hexdigest()[:16]


def index_meeting(
    transcript: str,
    meeting_id: str | None = None
) -> dict:

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript cannot be empty."
        )

    if meeting_id is None:
        meeting_id = create_meeting_id(
            transcript
        )

    chunks = chunk_transcript(
        transcript
    )

    indexed_count = index_chunks(
        chunks=chunks,
        meeting_id=meeting_id
    )

    return {
        "meeting_id": meeting_id,
        "chunks": chunks,
        "chunk_count": indexed_count
    }


def ask_meeting(
    question: str,
    meeting_id: str,
    top_k: int = 5
) -> dict:

    retrieved_chunks = retrieve_chunks(
        query=question,
        meeting_id=meeting_id,
        top_k=top_k
    )

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }