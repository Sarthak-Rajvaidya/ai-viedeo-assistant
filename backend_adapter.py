"""
backend_adapter.py
===============================================================================
Single integration point between the Streamlit frontend (app.py) and the
EXISTING AI backend (core/*, utils/*).

WHY THIS FILE EXISTS
-------------------------------------------------------------------------------
The exact function names / parameter shapes / return structures of the real
backend (analyzer.py, classifier.py, extractor.py, summarize.py,
transcriber.py, sarvam_transcriber.py, audio_processor.py) were not fully
known when this frontend was written. Rather than scatter guesses about
"what analyzer.py returns" all over app.py, every such assumption is
isolated HERE and clearly marked with `# ASSUMPTION:`.

>>> If your real backend uses different function/module names, or a
>>> different return dict shape, you only need to edit THIS file.
>>> app.py should almost never need to change.
===============================================================================
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


class BackendNotWiredError(Exception):
    """Raised when a required backend function could not be imported."""

    def __init__(self, what: str, detail: str = ""):
        msg = f"Backend function not available: {what}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ------------------------------------------------------------------------- #
# 1. IMPORTS FROM YOUR EXISTING BACKEND
# ------------------------------------------------------------------------- #
# ASSUMPTION: adjust these import lines to match your real module/function
# names. Each import is wrapped in try/except so the app can still boot (and
# surface a clear, non-crashing error in the UI) even if a module is missing
# or was renamed while you wire things up.

_IMPORT_ERRORS: dict[str, str] = {}


def _safe_import(label: str, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        _IMPORT_ERRORS[label] = str(e)
        return None


# utils/audio_processor.py
# ASSUMPTION: process_input(source: str) -> dict, where `source` is either a
# YouTube URL or a local file path, and the dict contains at least a WAV
# path and a list of chunk paths, e.g.:
#   {"wav_path": "...", "chunks": ["chunk1.wav", ...], "language": "hi"}
process_input = _safe_import(
    "utils.audio_processor.process_input",
    lambda: __import__("utils.audio_processor", fromlist=["process_input"]).process_input,
)

# core/transcriber.py (OpenAI Whisper path — English / general speech)
# ASSUMPTION: transcribe(audio_info: dict) -> str  (plain transcript text)
whisper_transcribe = _safe_import(
    "core.transcriber.transcribe",
    lambda: __import__("core.transcriber", fromlist=["transcribe"]).transcribe,
)

# core/sarvam_transcriber.py (Sarvam Saaras v3 — Indian-language speech -> English)
# ASSUMPTION: transcribe(audio_info: dict) -> str  (English translated transcript)
sarvam_transcribe = _safe_import(
    "core.sarvam_transcriber.transcribe",
    lambda: __import__("core.sarvam_transcriber", fromlist=["transcribe"]).transcribe,
)

# core/analyzer.py — the single orchestrating function, if you have it.
# ASSUMPTION: analyze_transcript(transcript: str) -> dict with (roughly):
#   {
#     "title": str,
#     "classification": {"content_type": str, "confidence": float},
#     "summary": {"overview": str, "key_points": [...], "takeaways": [...],
#                 "conclusions": str},
#     "action_items": [{"task": str, "owner": str, "deadline": str,
#                        "priority": str}, ...],
#     "key_decisions": [str, ...],
#     "open_questions": [str, ...],
#     "key_topics": [str, ...],
#   }
analyze_transcript = _safe_import(
    "core.analyzer.analyze_transcript",
    lambda: __import__("core.analyzer", fromlist=["analyze_transcript"]).analyze_transcript,
)

# Optional granular fallbacks, used only if analyze_transcript() is not
# available as a single orchestrating function.
classify_content = _safe_import(
    "core.classifier.classify_content",
    lambda: __import__("core.classifier", fromlist=["classify_content"]).classify_content,
)
summarize_transcript = _safe_import(
    "core.summarize.summarize",
    lambda: __import__("core.summarize", fromlist=["summarize"]).summarize,
)
extract_information = _safe_import(
    "core.extractor.extract_information",
    lambda: __import__("core.extractor", fromlist=["extract_information"]).extract_information,
)
generate_title = _safe_import(
    "core.analyzer.generate_title",
    lambda: __import__("core.analyzer", fromlist=["generate_title"]).generate_title,
)


def backend_health() -> dict[str, bool]:
    """Used by the sidebar 'System Status' panel."""
    return {
        "Audio Acquisition": process_input is not None,
        "Whisper Transcription": whisper_transcribe is not None,
        "Sarvam Translation": sarvam_transcribe is not None,
        "Mistral Analysis": (analyze_transcript is not None)
        or all([classify_content, summarize_transcript, extract_information]),
    }


def missing_backend_report() -> str:
    if not _IMPORT_ERRORS:
        return ""
    lines = ["Some backend modules could not be imported:"]
    for label, err in _IMPORT_ERRORS.items():
        lines.append(f"  • {label} -> {err}")
    lines.append(
        "Open backend_adapter.py and fix the import paths at the top of the file."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------------- #
# 2. NORMALIZED RESULT OBJECT
# ------------------------------------------------------------------------- #
# The rest of the app (app.py) only ever talks to this dataclass, never to
# raw backend dicts. That keeps app.py stable even if the backend's exact
# JSON shape changes.


@dataclass
class MeetingAnalysis:
    title: str = "Untitled"
    content_type: str = "unknown"
    confidence: float = 0.0
    transcript: str = ""
    language: str = "unknown"

    overview: str = ""
    key_points: list[str] = field(default_factory=list)
    takeaways: list[str] = field(default_factory=list)
    conclusions: str = ""

    action_items: list[dict[str, str]] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)

    raw: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

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


def _normalize_analysis_dict(raw: dict[str, Any], transcript: str) -> MeetingAnalysis:
    """
    ASSUMPTION LAYER: translates whatever shape analyze_transcript() (or the
    granular classify/summarize/extract functions) actually returns into the
    stable `MeetingAnalysis` object the UI relies on.

    Every `.get(...)` below with a fallback key represents a guess about your
    backend's naming. Adjust the key names here if needed — nothing else in
    the app has to change.
    """
    classification = raw.get("classification", raw.get("classify", {})) or {}
    summary = raw.get("summary", {}) or {}

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

    title = raw.get("title") or raw.get("generated_title") or "Untitled"

    overview = summary.get("overview") or summary.get("summary") or raw.get("summary_text") or ""
    key_points = summary.get("key_points") or summary.get("key_discussion_points") or []
    takeaways = summary.get("takeaways") or summary.get("main_takeaways") or []
    conclusions = summary.get("conclusions") or summary.get("conclusion") or ""

    action_items_raw = raw.get("action_items") or []
    action_items: list[dict[str, str]] = []
    for item in action_items_raw:
        if isinstance(item, dict):
            action_items.append(
                {
                    "task": str(item.get("task") or item.get("action") or ""),
                    "owner": str(item.get("owner") or "Not specified"),
                    "deadline": str(item.get("deadline") or "Not specified"),
                    "priority": str(item.get("priority") or "Not specified"),
                }
            )
        else:
            action_items.append(
                {"task": str(item), "owner": "Not specified",
                 "deadline": "Not specified", "priority": "Not specified"}
            )

    key_decisions = raw.get("key_decisions") or raw.get("decisions") or []
    open_questions = raw.get("open_questions") or raw.get("questions") or []
    key_topics = raw.get("key_topics") or raw.get("topics") or []

    return MeetingAnalysis(
        title=title,
        content_type=str(content_type).title(),
        confidence=float(confidence) if confidence else 0.0,
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


# ------------------------------------------------------------------------- #
# 3. PIPELINE STEP FUNCTIONS
# ------------------------------------------------------------------------- #
# Each function below wraps exactly ONE backend call. app.py calls these one
# at a time so it can update a real, non-fake progress indicator between
# each actual operation.


def acquire_and_preprocess_audio(source: str) -> dict[str, Any]:
    """
    Runs YouTube download / local file handling + preprocessing
    (WAV conversion, mono, 16kHz, chunking) via utils/audio_processor.py.
    """
    if process_input is None:
        raise BackendNotWiredError("utils.audio_processor.process_input")
    return process_input(source)


def transcribe(audio_info: dict[str, Any], mode: str = "auto") -> tuple[str, str]:
    """
    Runs speech-to-text / translation.

    mode:
      "auto"   -> ASSUMPTION: pick Sarvam if audio_info['language'] suggests
                  a non-English / Indian language, else Whisper.
      "whisper"-> force Whisper
      "sarvam" -> force Sarvam

    Returns: (transcript_text, engine_used)
    """
    detected_lang = str(audio_info.get("language", "")).lower()
    use_sarvam = mode == "sarvam" or (
        mode == "auto" and detected_lang not in ("", "en", "english")
    )

    if use_sarvam:
        if sarvam_transcribe is None:
            raise BackendNotWiredError("core.sarvam_transcriber.transcribe")
        text = sarvam_transcribe(audio_info)
        return text, "Sarvam Saaras v3"

    if whisper_transcribe is None:
        raise BackendNotWiredError("core.transcriber.transcribe")
    text = whisper_transcribe(audio_info)
    return text, "OpenAI Whisper"


def run_full_analysis(transcript: str) -> MeetingAnalysis:
    """
    Runs classification + title generation + summarization + structured
    extraction, preferring a single analyze_transcript() orchestrator if
    available, falling back to granular calls otherwise.
    """
    if analyze_transcript is not None:
        raw = analyze_transcript(transcript)
        if not isinstance(raw, dict):
            # ASSUMPTION: if analyze_transcript returns an object instead of
            # a dict, adjust this conversion (e.g. raw = raw.__dict__).
            raw = dict(raw)
        return _normalize_analysis_dict(raw, transcript)

    # --- granular fallback path ---
    if not all([classify_content, summarize_transcript, extract_information]):
        raise BackendNotWiredError(
            "core.analyzer.analyze_transcript (or classify_content + "
            "summarize + extract_information)"
        )

    classification = classify_content(transcript) or {}
    summary = summarize_transcript(transcript) or {}
    extraction = extract_information(transcript) or {}
    title = generate_title(transcript) if generate_title else extraction.get("title", "Untitled")

    raw = {
        "title": title,
        "classification": classification,
        "summary": summary,
        **extraction,
    }
    return _normalize_analysis_dict(raw, transcript)


def save_report(analysis: MeetingAnalysis, filename: str = "meeting_analysis.json") -> Path:
    """Persists the analysis JSON to reports/, as the existing backend does."""
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis.to_json_dict(), f, indent=2, ensure_ascii=False)
    return path


def safe_traceback() -> str:
    """Server-side-only traceback string; never render this directly to users."""
    return traceback.format_exc()