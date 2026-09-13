"""
backend_adapter.py
===============================================================================

Single integration point between the Streamlit frontend (app.py) and:

1. Existing AI backend
   - Audio acquisition
   - Whisper transcription
   - Sarvam transcription
   - Mistral analysis

2. RAG backend
   - HuggingFace embeddings
   - Qdrant Cloud
   - Groq generation

The Streamlit app should communicate with the backend only through
this adapter.

===============================================================================
"""

from __future__ import annotations

import json
import os
import traceback

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# REPORTS
# =============================================================================

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class BackendNotWiredError(Exception):
    """Raised when a required backend function could not be imported."""

    def __init__(self, what: str, detail: str = ""):
        msg = f"Backend function not available: {what}"

        if detail:
            msg += f" ({detail})"

        super().__init__(msg)


# =============================================================================
# IMPORT MANAGEMENT
# =============================================================================

_IMPORT_ERRORS: dict[str, str] = {}


def _safe_import(label: str, fn):
    """
    Safely import a backend function.

    If an import fails, the application can still start and
    display the problem through the System Status section.
    """

    try:
        return fn()

    except Exception as e:
        _IMPORT_ERRORS[label] = str(e)
        return None


# =============================================================================
# EXISTING BACKEND IMPORTS
# =============================================================================

# -----------------------------------------------------------------------------
# Audio acquisition / preprocessing
# -----------------------------------------------------------------------------

process_input = _safe_import(
    "utils.audio_processor.process_input",
    lambda: __import__(
        "utils.audio_processor",
        fromlist=["process_input"]
    ).process_input,
)


# -----------------------------------------------------------------------------
# Whisper transcription
# -----------------------------------------------------------------------------

whisper_transcribe = _safe_import(
    "transcriber.transcribe",
    lambda: __import__(
        "transcriber",
        fromlist=["transcribe"]
    ).transcribe,
)


# -----------------------------------------------------------------------------
# Sarvam transcription
# -----------------------------------------------------------------------------

sarvam_transcribe = _safe_import(
    "sarvam_transcriber.transcribe",
    lambda: __import__(
        "sarvam_transcriber",
        fromlist=["transcribe"]
    ).transcribe,
)


# -----------------------------------------------------------------------------
# Main Mistral analysis orchestrator
# -----------------------------------------------------------------------------

analyze_transcript = _safe_import(
    "core.analyzer.analyze_transcript",
    lambda: __import__(
        "core.analyzer",
        fromlist=["analyze_transcript"]
    ).analyze_transcript,
)


# -----------------------------------------------------------------------------
# Optional granular fallbacks
# -----------------------------------------------------------------------------

classify_content = _safe_import(
    "core.classifier.classify_content",
    lambda: __import__(
        "core.classifier",
        fromlist=["classify_content"]
    ).classify_content,
)


summarize_transcript = _safe_import(
    "core.summarize.summarize",
    lambda: __import__(
        "core.summarize",
        fromlist=["summarize"]
    ).summarize,
)


extract_information = _safe_import(
    "core.extractor.extract_information",
    lambda: __import__(
        "core.extractor",
        fromlist=["extract_information"]
    ).extract_information,
)


generate_title = _safe_import(
    "core.analyzer.generate_title",
    lambda: __import__(
        "core.analyzer",
        fromlist=["generate_title"]
    ).generate_title,
)


# =============================================================================
# RAG IMPORTS
# =============================================================================

# These functions come from:
#
# core/rag/pipeline.py
#
# index_meeting()
# ask_meeting()

rag_index_meeting = _safe_import(
    "core.rag.pipeline.index_meeting",
    lambda: __import__(
        "core.rag.pipeline",
        fromlist=["index_meeting"]
    ).index_meeting,
)


rag_ask_meeting = _safe_import(
    "core.rag.pipeline.ask_meeting",
    lambda: __import__(
        "core.rag.pipeline",
        fromlist=["ask_meeting"]
    ).ask_meeting,
)


# =============================================================================
# BACKEND HEALTH
# =============================================================================

def backend_health() -> dict[str, bool]:
    """
    Return the health status of all major backend components.
    """

    qdrant_ready = (
        rag_index_meeting is not None
        and rag_ask_meeting is not None
        and bool(os.getenv("QDRANT_URL"))
        and bool(os.getenv("QDRANT_API_KEY"))
    )

    groq_ready = (
        rag_ask_meeting is not None
        and bool(os.getenv("GROQ_API_KEY"))
    )

    return {
        "Audio Acquisition": process_input is not None,

        "Whisper Transcription":
            whisper_transcribe is not None,

        "Sarvam Translation":
            sarvam_transcribe is not None,

        "Mistral Analysis":
            (
                analyze_transcript is not None
                or all(
                    [
                        classify_content,
                        summarize_transcript,
                        extract_information,
                    ]
                )
            ),

        "Qdrant RAG": qdrant_ready,

        "Groq RAG": groq_ready,
    }


# =============================================================================
# IMPORT ERROR REPORT
# =============================================================================

def missing_backend_report() -> str:
    """
    Return readable backend import errors.
    """

    if not _IMPORT_ERRORS:
        return ""

    lines = [
        "Some backend modules could not be imported:"
    ]

    for label, err in _IMPORT_ERRORS.items():
        lines.append(
            f"  • {label} -> {err}"
        )

    lines.append(
        "Check the module/function paths in backend_adapter.py."
    )

    return "\n".join(lines)


# =============================================================================
# NORMALIZED MEETING ANALYSIS OBJECT
# =============================================================================

@dataclass
class MeetingAnalysis:

    title: str = "Untitled"

    content_type: str = "unknown"

    confidence: float = 0.0

    transcript: str = ""

    language: str = "unknown"

    overview: str = ""

    key_points: list[str] = field(
        default_factory=list
    )

    takeaways: list[str] = field(
        default_factory=list
    )

    conclusions: str = ""

    action_items: list[dict[str, str]] = field(
        default_factory=list
    )

    key_decisions: list[str] = field(
        default_factory=list
    )

    open_questions: list[str] = field(
        default_factory=list
    )

    key_topics: list[str] = field(
        default_factory=list
    )

    raw: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # -------------------------------------------------------------------------
    # JSON representation
    # -------------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:

        return {
            "title": self.title,

            "content_type": self.content_type,

            "confidence": self.confidence,

            "language": self.language,

            "created_at": self.created_at,

            "summary": {
                "overview": self.overview,

                "key_points": self.key_points,

                "takeaways": self.takeaways,

                "conclusions": self.conclusions,
            },

            "action_items": self.action_items,

            "key_decisions": self.key_decisions,

            "open_questions": self.open_questions,

            "key_topics": self.key_topics,

            "transcript": self.transcript,
        }


# =============================================================================
# NORMALIZE ANALYSIS
# =============================================================================

def _normalize_analysis_dict(
    raw: dict[str, Any],
    transcript: str,
) -> MeetingAnalysis:

    classification = (
        raw.get(
            "classification",
            raw.get("classify", {})
        )
        or {}
    )

    summary = (
        raw.get("summary", {})
        or {}
    )

    content_type = (
        classification.get("content_type")
        or classification.get("label")
        or raw.get("content_type")
        or "unknown"
    )

    confidence = (
        classification.get("confidence")
        or raw.get("confidence")
        or 0.0
    )

    title = (
        raw.get("title")
        or raw.get("generated_title")
        or "Untitled"
    )

    overview = (
        summary.get("overview")
        or summary.get("summary")
        or raw.get("summary_text")
        or ""
    )

    key_points = (
        summary.get("key_points")
        or summary.get("key_discussion_points")
        or []
    )

    takeaways = (
        summary.get("takeaways")
        or summary.get("main_takeaways")
        or []
    )

    conclusions = (
        summary.get("conclusions")
        or summary.get("conclusion")
        or ""
    )

    # -------------------------------------------------------------------------
    # Action items
    # -------------------------------------------------------------------------

    action_items_raw = (
        raw.get("action_items")
        or []
    )

    action_items: list[dict[str, str]] = []

    for item in action_items_raw:

        if isinstance(item, dict):

            action_items.append(
                {
                    "task": str(
                        item.get("task")
                        or item.get("action")
                        or ""
                    ),

                    "owner": str(
                        item.get("owner")
                        or "Not specified"
                    ),

                    "deadline": str(
                        item.get("deadline")
                        or "Not specified"
                    ),

                    "priority": str(
                        item.get("priority")
                        or "Not specified"
                    ),
                }
            )

        else:

            action_items.append(
                {
                    "task": str(item),

                    "owner": "Not specified",

                    "deadline": "Not specified",

                    "priority": "Not specified",
                }
            )

    # -------------------------------------------------------------------------
    # Other structured information
    # -------------------------------------------------------------------------

    key_decisions = (
        raw.get("key_decisions")
        or raw.get("decisions")
        or []
    )

    open_questions = (
        raw.get("open_questions")
        or raw.get("questions")
        or []
    )

    key_topics = (
        raw.get("key_topics")
        or raw.get("topics")
        or []
    )

    return MeetingAnalysis(

        title=title,

        content_type=str(
            content_type
        ).title(),

        confidence=(
            float(confidence)
            if confidence
            else 0.0
        ),

        transcript=transcript,

        overview=overview,

        key_points=list(key_points),

        takeaways=list(takeaways),

        conclusions=conclusions,

        action_items=action_items,

        key_decisions=list(key_decisions),

        open_questions=list(open_questions),

        key_topics=list(key_topics),

        raw=raw,
    )


# =============================================================================
# AUDIO
# =============================================================================

def acquire_and_preprocess_audio(
    source: str,
) -> dict[str, Any]:

    if process_input is None:

        raise BackendNotWiredError(
            "utils.audio_processor.process_input"
        )

    return process_input(source)


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe(
    audio_info: dict[str, Any],
    mode: str = "auto",
) -> tuple[str, str]:

    detected_lang = str(
        audio_info.get(
            "language",
            ""
        )
    ).lower()

    use_sarvam = (
        mode == "sarvam"
        or (
            mode == "auto"
            and detected_lang not in (
                "",
                "en",
                "english",
            )
        )
    )

    # -------------------------------------------------------------------------
    # Sarvam
    # -------------------------------------------------------------------------

    if use_sarvam:

        if sarvam_transcribe is None:

            raise BackendNotWiredError(
                "sarvam_transcriber.transcribe"
            )

        text = sarvam_transcribe(
            audio_info
        )

        return text, "Sarvam Saaras v3"

    # -------------------------------------------------------------------------
    # Whisper
    # -------------------------------------------------------------------------

    if whisper_transcribe is None:

        raise BackendNotWiredError(
            "core.transcriber.transcribe"
        )

    text = whisper_transcribe(
        audio_info
    )

    return text, "OpenAI Whisper"


# =============================================================================
# FULL AI ANALYSIS
# =============================================================================

def run_full_analysis(
    transcript: str,
) -> MeetingAnalysis:

    # -------------------------------------------------------------------------
    # Preferred orchestrator
    # -------------------------------------------------------------------------

    if analyze_transcript is not None:

        raw = analyze_transcript(
            transcript
        )

        if not isinstance(raw, dict):

            raw = dict(raw)

        return _normalize_analysis_dict(
            raw,
            transcript,
        )

    # -------------------------------------------------------------------------
    # Granular fallback
    # -------------------------------------------------------------------------

    if not all(
        [
            classify_content,
            summarize_transcript,
            extract_information,
        ]
    ):

        raise BackendNotWiredError(
            "core.analyzer.analyze_transcript "
            "(or classify_content + summarize + "
            "extract_information)"
        )

    classification = (
        classify_content(transcript)
        or {}
    )

    summary = (
        summarize_transcript(transcript)
        or {}
    )

    extraction = (
        extract_information(transcript)
        or {}
    )

    title = (
        generate_title(transcript)
        if generate_title
        else extraction.get(
            "title",
            "Untitled"
        )
    )

    raw = {
        "title": title,

        "classification": classification,

        "summary": summary,

        **extraction,
    }

    return _normalize_analysis_dict(
        raw,
        transcript,
    )


# =============================================================================
# RAG — INDEX MEETING
# =============================================================================

def index_transcript_for_rag(
    transcript: str,
    meeting_id: str | None = None,
) -> dict[str, Any]:
    """
    Index the actual meeting transcript into:

    Transcript
        ↓
    Chunking
        ↓
    HuggingFace embeddings
        ↓
    Qdrant Cloud
    """

    if rag_index_meeting is None:

        raise BackendNotWiredError(
            "core.rag.pipeline.index_meeting"
        )

    return rag_index_meeting(
        transcript=transcript,
        meeting_id=meeting_id,
    )


# =============================================================================
# RAG — CHAT WITH MEETING
# =============================================================================

def chat_with_meeting(
    question: str,
    meeting_id: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Retrieve relevant meeting chunks and generate
    an answer using Groq.
    """

    if rag_ask_meeting is None:

        raise BackendNotWiredError(
            "core.rag.pipeline.ask_meeting"
        )

    return rag_ask_meeting(
        question=question,
        meeting_id=meeting_id,
        top_k=top_k,
    )


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_report(
    analysis: MeetingAnalysis,
    filename: str = "meeting_analysis.json",
) -> Path:

    path = REPORTS_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            analysis.to_json_dict(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return path


# =============================================================================
# SAFE TRACEBACK
# =============================================================================

def safe_traceback() -> str:
    """
    Return traceback for server-side logging only.
    """

    return traceback.format_exc()