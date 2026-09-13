"""
backend_adapter.py
===============================================================================
Integration layer between Streamlit frontend and the existing AI backend.

Architecture:

    Streamlit app.py
          |
          v
    backend_adapter.py
          |
          +--> Audio Acquisition
          +--> Whisper
          +--> Sarvam
          +--> Mistral Analysis
          +--> HuggingFace Embeddings
          +--> Qdrant Cloud
          +--> Groq LLM
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

from dotenv import load_dotenv


# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()


# =============================================================================
# DIRECTORIES
# =============================================================================

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# CUSTOM ERROR
# =============================================================================

class BackendNotWiredError(Exception):
    """Raised when a required backend component is unavailable."""

    def __init__(self, what: str, detail: str = ""):

        message = f"Backend function not available: {what}"

        if detail:
            message += f" ({detail})"

        super().__init__(message)


# =============================================================================
# IMPORT ERROR STORAGE
# =============================================================================

_IMPORT_ERRORS: dict[str, str] = {}


def _safe_import(label: str, fn):

    try:

        return fn()

    except Exception as e:

        _IMPORT_ERRORS[label] = str(e)

        return None


# =============================================================================
# AUDIO PROCESSOR
# =============================================================================

process_input = _safe_import(
    "utils.audio_processor.process_input",

    lambda: __import__(
        "utils.audio_processor",
        fromlist=["process_input"]
    ).process_input
)


# =============================================================================
# WHISPER
# =============================================================================

"""
Actual project function:

core/transcriber.py

    transcribe_chunk()
    transcribe_all()

There is no transcribe() function.
"""

whisper_transcribe_all = _safe_import(
    "core.transcriber.transcribe_all",

    lambda: __import__(
        "core.transcriber",
        fromlist=["transcribe_all"]
    ).transcribe_all
)


# =============================================================================
# SARVAM
# =============================================================================

"""
Actual project function:

core/sarvam_transcriber.py

    transcribe_sarvam_batch()
"""

sarvam_transcribe_batch = _safe_import(
    "core.sarvam_transcriber.transcribe_sarvam_batch",

    lambda: __import__(
        "core.sarvam_transcriber",
        fromlist=["transcribe_sarvam_batch"]
    ).transcribe_sarvam_batch
)


# =============================================================================
# MISTRAL ANALYSIS
# =============================================================================

analyze_transcript = _safe_import(
    "core.analyzer.analyze_transcript",

    lambda: __import__(
        "core.analyzer",
        fromlist=["analyze_transcript"]
    ).analyze_transcript
)


# =============================================================================
# OPTIONAL ANALYSIS FUNCTIONS
# =============================================================================

classify_content = _safe_import(
    "core.classifier.classify_content",

    lambda: __import__(
        "core.classifier",
        fromlist=["classify_content"]
    ).classify_content
)


summarize_transcript = _safe_import(
    "core.summarize.summarize",

    lambda: __import__(
        "core.summarize",
        fromlist=["summarize"]
    ).summarize
)


extract_information = _safe_import(
    "core.extractor.extract_information",

    lambda: __import__(
        "core.extractor",
        fromlist=["extract_information"]
    ).extract_information
)


generate_title = _safe_import(
    "core.analyzer.generate_title",

    lambda: __import__(
        "core.analyzer",
        fromlist=["generate_title"]
    ).generate_title
)


# =============================================================================
# RAG IMPORTS
# =============================================================================

"""
IMPORTANT:

Do NOT import the entire RAG pipeline just to check Qdrant health.

We separately import:

    index_meeting()
    ask_meeting()

This makes failures much easier to diagnose.
"""

index_meeting = _safe_import(
    "core.rag.pipeline.index_meeting",

    lambda: __import__(
        "core.rag.pipeline",
        fromlist=["index_meeting"]
    ).index_meeting
)


ask_meeting = _safe_import(
    "core.rag.pipeline.ask_meeting",

    lambda: __import__(
        "core.rag.pipeline",
        fromlist=["ask_meeting"]
    ).ask_meeting
)


# =============================================================================
# RAG / QDRANT CONFIGURATION CHECK
# =============================================================================

def qdrant_configured() -> bool:

    url = os.getenv("QDRANT_URL")

    api_key = os.getenv("QDRANT_API_KEY")

    collection = os.getenv(
        "QDRANT_COLLECTION",
        "meeting_transcripts"
    )

    return bool(
        url
        and url.strip()
        and api_key
        and api_key.strip()
        and collection
        and collection.strip()
    )


# =============================================================================
# GROQ CONFIGURATION CHECK
# =============================================================================

def groq_configured() -> bool:

    api_key = os.getenv("GROQ_API_KEY")

    return bool(
        api_key
        and api_key.strip()
    )


# =============================================================================
# BACKEND HEALTH
# =============================================================================

def backend_health() -> dict[str, bool]:

    """
    Returns the health/availability status of backend components.

    For external services:

        Qdrant Cloud -> checks configuration + RAG wiring
        Groq          -> checks API key + RAG generation wiring
    """

    # -------------------------------------------------------------------------
    # QDRANT
    # -------------------------------------------------------------------------

    qdrant_ok = (
        qdrant_configured()
        and index_meeting is not None
        and ask_meeting is not None
    )

    # -------------------------------------------------------------------------
    # GROQ
    # -------------------------------------------------------------------------

    groq_generator_imported = _safe_import(
        "core.rag.generator.generate_answer",

        lambda: __import__(
            "core.rag.generator",
            fromlist=["generate_answer"]
        ).generate_answer
    )

    groq_ok = (
        groq_configured()
        and groq_generator_imported is not None
    )

    # -------------------------------------------------------------------------
    # MISTRAL
    # -------------------------------------------------------------------------

    mistral_ok = (
        analyze_transcript is not None
        or all([
            classify_content is not None,
            summarize_transcript is not None,
            extract_information is not None
        ])
    )

    return {

        "Audio Acquisition":
            process_input is not None,

        "Whisper Transcription":
            whisper_transcribe_all is not None,

        "Sarvam Translation":
            sarvam_transcribe_batch is not None,

        "Mistral AI":
            mistral_ok,

        "Qdrant Cloud":
            qdrant_ok,

        "Groq LLM":
            groq_ok,
    }


# =============================================================================
# DETAILED HEALTH INFORMATION
# =============================================================================

def backend_diagnostics() -> dict[str, Any]:

    health = backend_health()

    return {
        "health": health,

        "qdrant_url_configured":
            bool(os.getenv("QDRANT_URL")),

        "qdrant_api_key_configured":
            bool(os.getenv("QDRANT_API_KEY")),

        "qdrant_collection":
            os.getenv(
                "QDRANT_COLLECTION",
                "meeting_transcripts"
            ),

        "groq_api_key_configured":
            bool(os.getenv("GROQ_API_KEY")),

        "import_errors":
            dict(_IMPORT_ERRORS),
    }


# =============================================================================
# IMPORT ERROR REPORT
# =============================================================================

def missing_backend_report() -> str:

    if not _IMPORT_ERRORS:

        return ""

    lines = [
        "Backend diagnostics:"
    ]

    for label, error in _IMPORT_ERRORS.items():

        lines.append(
            f"  • {label} -> {error}"
        )

    return "\n".join(lines)


# =============================================================================
# MEETING ANALYSIS
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

    def to_json_dict(self) -> dict[str, Any]:

        return {

            "title":
                self.title,

            "content_type":
                self.content_type,

            "confidence":
                self.confidence,

            "language":
                self.language,

            "created_at":
                self.created_at,

            "summary": {

                "overview":
                    self.overview,

                "key_points":
                    self.key_points,

                "takeaways":
                    self.takeaways,

                "conclusions":
                    self.conclusions,
            },

            "action_items":
                self.action_items,

            "key_decisions":
                self.key_decisions,

            "open_questions":
                self.open_questions,

            "key_topics":
                self.key_topics,

            "transcript":
                self.transcript,
        }


# =============================================================================
# NORMALIZE ANALYSIS
# =============================================================================

def _normalize_analysis_dict(
    raw: dict[str, Any],
    transcript: str
) -> MeetingAnalysis:

    classification = (
        raw.get("classification")
        or raw.get("classify")
        or {}
    )

    summary = (
        raw.get("summary")
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
    # ACTION ITEMS
    # -------------------------------------------------------------------------

    action_items_raw = (
        raw.get("action_items")
        or []
    )

    action_items = []

    for item in action_items_raw:

        if isinstance(item, dict):

            action_items.append({

                "task":
                    str(
                        item.get("task")
                        or item.get("action")
                        or ""
                    ),

                "owner":
                    str(
                        item.get("owner")
                        or "Not specified"
                    ),

                "deadline":
                    str(
                        item.get("deadline")
                        or "Not specified"
                    ),

                "priority":
                    str(
                        item.get("priority")
                        or "Not specified"
                    ),
            })

        else:

            action_items.append({

                "task":
                    str(item),

                "owner":
                    "Not specified",

                "deadline":
                    "Not specified",

                "priority":
                    "Not specified",
            })

    # -------------------------------------------------------------------------
    # OTHER FIELDS
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

        key_points=list(
            key_points
        ),

        takeaways=list(
            takeaways
        ),

        conclusions=conclusions,

        action_items=action_items,

        key_decisions=list(
            key_decisions
        ),

        open_questions=list(
            open_questions
        ),

        key_topics=list(
            key_topics
        ),

        raw=raw,
    )


# =============================================================================
# AUDIO ACQUISITION
# =============================================================================

def acquire_and_preprocess_audio(
    source: str
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
    mode: str = "auto"
) -> tuple[str, str]:

    if not audio_info:

        raise ValueError(
            "Audio information is empty."
        )

    detected_lang = str(
        audio_info.get("language", "")
    ).lower()

    # -------------------------------------------------------------------------
    # LANGUAGE DECISION
    # -------------------------------------------------------------------------

    use_sarvam = (

        mode == "sarvam"

        or (

            mode == "auto"

            and detected_lang not in (
                "",
                "en",
                "english"
            )
        )
    )

    # =========================================================================
    # SARVAM
    # =========================================================================

    if use_sarvam:

        if sarvam_transcribe_batch is None:

            raise BackendNotWiredError(
                "core.sarvam_transcriber.transcribe_sarvam_batch"
            )

        wav_path = (

            audio_info.get("wav_path")

            or audio_info.get("audio_path")

            or audio_info.get("path")
        )

        if not wav_path:

            raise ValueError(
                "Could not find WAV/audio path for Sarvam transcription."
            )

        text = sarvam_transcribe_batch(
            wav_path
        )

        return text, "Sarvam Saaras v3"

    # =========================================================================
    # WHISPER
    # =========================================================================

    if whisper_transcribe_all is None:

        raise BackendNotWiredError(
            "core.transcriber.transcribe_all"
        )

    chunks = (
        audio_info.get("chunks")
        or []
    )

    if not chunks:

        wav_path = (

            audio_info.get("wav_path")

            or audio_info.get("audio_path")

            or audio_info.get("path")
        )

        if not wav_path:

            raise ValueError(
                "No audio chunks or WAV path found."
            )

        chunks = [wav_path]

    text = whisper_transcribe_all(
        chunks,
        translate=False
    )

    return text, "OpenAI Whisper"


# =============================================================================
# FULL AI ANALYSIS
# =============================================================================

def run_full_analysis(
    transcript: str
) -> MeetingAnalysis:

    if not transcript or not transcript.strip():

        raise ValueError(
            "Transcript is empty."
        )

    # -------------------------------------------------------------------------
    # PREFERRED ANALYZER
    # -------------------------------------------------------------------------

    if analyze_transcript is not None:

        raw = analyze_transcript(
            transcript
        )

        if not isinstance(raw, dict):

            raw = dict(raw)

        return _normalize_analysis_dict(
            raw,
            transcript
        )

    # -------------------------------------------------------------------------
    # FALLBACK
    # -------------------------------------------------------------------------

    if not all([
        classify_content,
        summarize_transcript,
        extract_information
    ]):

        raise BackendNotWiredError(
            "core.analyzer.analyze_transcript"
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

        "title":
            title,

        "classification":
            classification,

        "summary":
            summary,

        **extraction,
    }

    return _normalize_analysis_dict(
        raw,
        transcript
    )


# =============================================================================
# RAG INDEXING
# =============================================================================

def index_transcript_for_rag(
    transcript: str,
    meeting_id: str | None = None
) -> dict[str, Any]:

    if index_meeting is None:

        raise BackendNotWiredError(
            "core.rag.pipeline.index_meeting"
        )

    if not transcript or not transcript.strip():

        raise ValueError(
            "Transcript cannot be empty."
        )

    return index_meeting(
        transcript=transcript,
        meeting_id=meeting_id
    )


# =============================================================================
# RAG CHAT
# =============================================================================

def chat_with_meeting(
    question: str,
    meeting_id: str,
    top_k: int = 5
) -> dict[str, Any]:

    if ask_meeting is None:

        raise BackendNotWiredError(
            "core.rag.pipeline.ask_meeting"
        )

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not meeting_id:

        raise ValueError(
            "No meeting is currently indexed."
        )

    return ask_meeting(
        question=question,
        meeting_id=meeting_id,
        top_k=top_k
    )


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_report(
    analysis: MeetingAnalysis,
    filename: str = "meeting_analysis.json"
) -> Path:

    path = REPORTS_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis.to_json_dict(),
            f,
            indent=2,
            ensure_ascii=False
        )

    return path


# =============================================================================
# SAFE TRACEBACK
# =============================================================================

def safe_traceback() -> str:

    return traceback.format_exc()