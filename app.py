"""
app.py
===============================================================================

AI Meeting Intelligence Assistant

Streamlit presentation / orchestration layer.

Complete pipeline:

    YouTube / Audio / Video
            ↓
    Audio Acquisition + Preprocessing
            ↓
    Whisper / Sarvam
            ↓
    Transcript
            ↓
    Mistral AI Analysis
            ↓
    JSON Report
            ↓
    HuggingFace Embeddings
            ↓
    Qdrant Cloud
            ↓
    RAG Retrieval
            ↓
    Groq LLM
            ↓
    Chat with Meeting

Run:

    python -m streamlit run app.py

===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from backend_adapter import (
    MeetingAnalysis,
    BackendNotWiredError,

    # Existing backend
    acquire_and_preprocess_audio,
    transcribe,
    run_full_analysis,
    save_report,

    # Health / diagnostics
    backend_health,
    missing_backend_report,
    safe_traceback,

    # RAG
    index_transcript_for_rag,
    chat_with_meeting,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Meeting Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_FORMATS = [
    "mp3",
    "wav",
    "m4a",
    "mp4",
    "webm",
    "mov",
    "mkv",
    "aac",
    "ogg",
]


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        .app-subtitle {
            color: var(--text-color-secondary, #8a8f98);
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.6rem;
        }

        .section-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }

        .pill {
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(99,102,241,0.15);
            color: #6366f1;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
        }

        .muted {
            color: var(--text-color-secondary, #8a8f98);
            font-size: 0.9rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 10px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_state() -> None:

    defaults = {

        # ---------------------------------------------------------------------
        # Existing application state
        # ---------------------------------------------------------------------

        "analysis": None,

        "processing": False,

        "last_error": None,

        "active_page": "Analyze",

        "report_path": None,

        "engine_used": None,

        # ---------------------------------------------------------------------
        # RAG state
        # ---------------------------------------------------------------------

        "meeting_id": None,

        "rag_indexed": False,

        "rag_chunk_count": 0,

        "chat_messages": [],

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


init_state()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        "### 🧠 AI Meeting Intelligence"
    )

    st.caption(
        "Meetings, lectures & videos → actionable intelligence"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    pages = [
        "Analyze",
        "Results",
        "Transcript",
        "Chat with Meeting",
    ]

    # Handle old session state from previous version
    if (
        st.session_state.active_page
        == "Chat with Meeting (Coming Soon)"
    ):

        st.session_state.active_page = (
            "Chat with Meeting"
        )

    if st.session_state.active_page not in pages:

        st.session_state.active_page = "Analyze"

    st.session_state.active_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.active_page
        ),
        label_visibility="collapsed",
    )

    st.divider()

    # -------------------------------------------------------------------------
    # System Status
    # -------------------------------------------------------------------------

    st.markdown(
        "#### System Status"
    )

    health = backend_health()

    # IMPORTANT:
    # These keys MUST match backend_adapter.backend_health()

    status_labels = {

        "Audio Acquisition":
            "Audio Acquisition",

        "Whisper Transcription":
            "Speech Recognition",

        "Sarvam Translation":
            "Sarvam Translation",

        "Mistral AI":
            "Mistral AI",

        "Qdrant Cloud":
            "Qdrant Cloud",

        "Groq LLM":
            "Groq LLM",

    }

    for key, label in status_labels.items():

        ok = health.get(
            key,
            False
        )

        st.markdown(
            f"{'🟢' if ok else '🔴'} {label}"
        )

    # -------------------------------------------------------------------------
    # RAG index status
    # -------------------------------------------------------------------------

    st.markdown("")

    if st.session_state.rag_indexed:

        st.markdown(
            "🟢 **RAG Index Ready**"
        )

        st.caption(
            f"{st.session_state.rag_chunk_count} "
            "transcript chunk(s) indexed in Qdrant Cloud"
        )

    else:

        st.markdown(
            "⚪ **RAG Index — Not Ready**"
        )

        st.caption(
            "Analyze a meeting to create its semantic index."
        )

    # -------------------------------------------------------------------------
    # Backend wiring issues
    # -------------------------------------------------------------------------

    if not all(health.values()):

        with st.expander(
            "⚠ Backend wiring issues"
        ):

            st.code(
                missing_backend_report()
                or "Check API keys and dependencies."
            )

    st.divider()

    st.caption(
        "v2.0 · Qdrant Cloud + HuggingFace + Groq"
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    "## AI Meeting Intelligence Assistant"
)

st.markdown(
    '<div class="app-subtitle">'
    "Transform meetings, lectures and videos into "
    "actionable intelligence."
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# WELCOME STATE
# =============================================================================

def render_welcome_state() -> None:

    st.markdown(
        '<div class="section-card">'
        "<h4>Turn hours of audio into minutes of intelligence.</h4>"
        '<p class="muted">'
        "Paste a YouTube link or upload a recording to get started."
        "</p>"
        "<ul>"
        "<li>✓ Multilingual speech processing</li>"
        "<li>✓ AI-generated summaries</li>"
        "<li>✓ Action item extraction</li>"
        "<li>✓ Decision & open-question detection</li>"
        "<li>✓ Topic extraction</li>"
        "<li>✓ Semantic meeting search</li>"
        "<li>✓ Chat with your meeting using RAG</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(source: str) -> None:

    """
    Complete processing pipeline.

    Existing AI pipeline:

        Audio
          ↓
        Whisper / Sarvam
          ↓
        Transcript
          ↓
        Mistral
          ↓
        MeetingAnalysis
          ↓
        JSON report

    RAG pipeline:

        Transcript
          ↓
        Chunking
          ↓
        HuggingFace embeddings
          ↓
        Qdrant Cloud
          ↓
        Ready for Groq chat
    """

    st.session_state.last_error = None

    try:

        with st.status(
            "Running AI pipeline…",
            expanded=True,
        ) as status:

            # ================================================================
            # STEP 1 — AUDIO ACQUISITION
            # ================================================================

            status.write(
                "🔄 Acquiring audio…"
            )

            audio_info = (
                acquire_and_preprocess_audio(
                    source
                )
            )

            status.write(
                "✅ Audio acquired"
            )

            # ================================================================
            # STEP 2 — PREPROCESSING
            # ================================================================

            status.write(
                "🔄 Preprocessing audio "
                "(mono · 16kHz · chunking)…"
            )

            # Audio preprocessing is handled inside
            # acquire_and_preprocess_audio()

            status.write(
                "✅ Audio preprocessed"
            )

            # ================================================================
            # STEP 3 — TRANSCRIPTION
            # ================================================================

            status.write(
                "🔄 Running speech recognition / translation…"
            )

            transcript_raw, engine = transcribe(
                audio_info,
                mode="auto",
            )

            st.session_state.engine_used = engine

            status.write(
                f"✅ Transcribed using {engine}"
            )

            if (
                not transcript_raw
                or not transcript_raw.strip()
            ):

                raise ValueError(
                    "Transcription returned an empty transcript."
                )

            # ================================================================
            # STEP 4 — AI ANALYSIS
            # ================================================================

            status.write(
                "🔄 Cleaning transcript…"
            )

            # Your existing analyzer handles
            # transcript processing/cleaning.

            status.write(
                "✅ Transcript cleaned"
            )

            status.write(
                "🔄 Running meeting analysis…"
            )

            analysis = run_full_analysis(
                transcript_raw
            )

            status.write(
                "✅ Analysis complete"
            )

            # ================================================================
            # STEP 5 — SAVE REPORT
            # ================================================================

            status.write(
                "🔄 Saving analysis report…"
            )

            report_path = save_report(
                analysis
            )

            st.session_state.report_path = (
                str(report_path)
            )

            status.write(
                f"✅ Saved to {report_path}"
            )

            # ================================================================
            # STEP 6 — RAG INDEXING
            # ================================================================

            status.write(
                "🔄 Creating semantic meeting index…"
            )

            rag_result = index_transcript_for_rag(
                transcript=analysis.transcript
            )

            # ---------------------------------------------------------------
            # Store RAG information
            # ---------------------------------------------------------------

            st.session_state.meeting_id = (
                rag_result["meeting_id"]
            )

            st.session_state.rag_indexed = True

            st.session_state.rag_chunk_count = (
                rag_result["chunk_count"]
            )

            # ---------------------------------------------------------------
            # New meeting = new conversation
            # ---------------------------------------------------------------

            st.session_state.chat_messages = []

            status.write(
                "✅ Transcript indexed in Qdrant Cloud"
            )

            status.write(
                f"✅ {rag_result['chunk_count']} "
                "semantic chunk(s) created"
            )

            # ================================================================
            # COMPLETE
            # ================================================================

            status.update(
                label="Pipeline complete",
                state="complete",
                expanded=False,
            )

        # ---------------------------------------------------------------------
        # Save final application state
        # ---------------------------------------------------------------------

        st.session_state.analysis = analysis

        st.session_state.active_page = "Results"

        st.success(
            "Analysis and RAG indexing completed successfully."
        )

        st.rerun()

    # =========================================================================
    # BACKEND ERROR
    # =========================================================================

    except BackendNotWiredError as e:

        st.session_state.last_error = str(e)

        st.error(
            "This feature isn't connected to the backend yet.\n\n"
            f"{e}"
        )

    # =========================================================================
    # VALUE ERROR
    # =========================================================================

    except ValueError as e:

        st.session_state.last_error = str(e)

        st.warning(
            str(e)
        )

    # =========================================================================
    # UNEXPECTED ERROR
    # =========================================================================

    except Exception:

        st.session_state.last_error = (
            "unexpected_error"
        )

        # Print full traceback only to terminal
        print(
            safe_traceback()
        )

        st.error(
            "Something went wrong while processing "
            "this content. Please check your input, "
            "API keys and backend configuration."
        )


# =============================================================================
# ANALYZE PAGE
# =============================================================================

def render_analyze_page() -> None:

    if st.session_state.analysis is None:

        render_welcome_state()

    st.markdown(
        "#### New Analysis"
    )

    tab_yt, tab_file = st.tabs(
        [
            "🔗 YouTube URL",
            "📁 Upload File",
        ]
    )

    source: Optional[str] = None

    # =========================================================================
    # YOUTUBE
    # =========================================================================

    with tab_yt:

        yt_url = st.text_input(
            "Paste YouTube URL",
            placeholder=(
                "https://youtube.com/watch?v=..."
            ),
            key="yt_url_input",
        )

        if yt_url:

            source = yt_url.strip()

    # =========================================================================
    # LOCAL FILE
    # =========================================================================

    with tab_file:

        uploaded = st.file_uploader(
            "Upload audio or video file",
            type=SUPPORTED_FORMATS,
            key="file_uploader",
        )

        if uploaded is not None:

            uploads_dir = Path(
                "uploads"
            )

            uploads_dir.mkdir(
                exist_ok=True
            )

            # Prevent path traversal from uploaded filename
            safe_filename = Path(
                uploaded.name
            ).name

            temp_path = (
                uploads_dir
                / safe_filename
            )

            with open(
                temp_path,
                "wb",
            ) as f:

                f.write(
                    uploaded.getbuffer()
                )

            st.caption(
                f"Ready: {uploaded.name} "
                f"({uploaded.size / 1_000_000:.1f} MB)"
            )

            source = str(
                temp_path
            )

    # =========================================================================
    # ANALYZE BUTTON
    # =========================================================================

    col_a, col_b = st.columns(
        [1, 4]
    )

    with col_a:

        analyze_clicked = st.button(
            "🚀 Analyze Content",
            type="primary",
            use_container_width=True,
        )

    if analyze_clicked:

        if not source:

            st.warning(
                "Please paste a YouTube URL "
                "or upload a file first."
            )

        elif (
            source.startswith("http")
            and "youtu" not in source.lower()
        ):

            st.warning(
                "That doesn't look like a valid YouTube URL."
            )

        else:

            run_pipeline(
                source
            )


# =============================================================================
# METRICS
# =============================================================================

def render_metrics(
    analysis: MeetingAnalysis,
) -> None:

    c1, c2, c3, c4 = st.columns(4)

    # -------------------------------------------------------------------------
    # Content Type
    # -------------------------------------------------------------------------

    c1.metric(
        "Content Type",
        analysis.content_type,
    )

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    confidence_display = (

        f"{analysis.confidence * 100:.0f}%"

        if analysis.confidence <= 1

        else f"{analysis.confidence:.0f}%"
    )

    c2.metric(
        "Confidence",
        confidence_display,
    )

    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    c3.metric(
        "Processing Status",
        "Complete ✅",
    )

    # -------------------------------------------------------------------------
    # Transcript length
    # -------------------------------------------------------------------------

    word_count = len(
        analysis.transcript.split()
    )

    c4.metric(
        "Transcript Length",
        f"{word_count:,} words",
    )


# =============================================================================
# RESULTS PAGE
# =============================================================================

def render_results_page() -> None:

    analysis: Optional[
        MeetingAnalysis
    ] = st.session_state.analysis

    if analysis is None:

        st.info(
            "No analysis yet. Go to **Analyze** "
            "to process a video or recording."
        )

        return

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    render_metrics(
        analysis
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    st.markdown(
        f"### {analysis.title}"
    )

    # -------------------------------------------------------------------------
    # Engine
    # -------------------------------------------------------------------------

    if st.session_state.engine_used:

        st.markdown(
            f'<span class="pill">'
            f'{st.session_state.engine_used}'
            f"</span>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # RAG status
    # -------------------------------------------------------------------------

    if st.session_state.rag_indexed:

        st.success(
            "🧠 RAG Ready — this meeting is indexed "
            "in Qdrant Cloud and can be queried."
        )

    # -------------------------------------------------------------------------
    # Result tabs
    # -------------------------------------------------------------------------

    tabs = st.tabs(
        [
            "📝 Summary",
            "✅ Action Items",
            "📌 Decisions",
            "❓ Questions",
            "🏷️ Topics",
            "⬇️ Downloads",
        ]
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    with tabs[0]:

        if analysis.overview:

            st.markdown(
                "**Overview**"
            )

            st.markdown(
                analysis.overview
            )

        if analysis.key_points:

            st.markdown(
                "**Key Discussion Points**"
            )

            for point in analysis.key_points:

                st.markdown(
                    f"- {point}"
                )

        if analysis.takeaways:

            st.markdown(
                "**Main Takeaways**"
            )

            for takeaway in analysis.takeaways:

                st.markdown(
                    f"- {takeaway}"
                )

        if analysis.conclusions:

            st.markdown(
                "**Conclusions**"
            )

            st.markdown(
                analysis.conclusions
            )

        if not any(
            [
                analysis.overview,
                analysis.key_points,
                analysis.takeaways,
                analysis.conclusions,
            ]
        ):

            st.info(
                "No summary available."
            )

    # =========================================================================
    # ACTION ITEMS
    # =========================================================================

    with tabs[1]:

        if analysis.action_items:

            st.dataframe(

                [
                    {
                        "Task":
                            item.get(
                                "task",
                                ""
                            ),

                        "Owner":
                            item.get(
                                "owner",
                                "Not specified"
                            ),

                        "Deadline":
                            item.get(
                                "deadline",
                                "Not specified"
                            ),

                        "Priority":
                            item.get(
                                "priority",
                                "Not specified"
                            ),
                    }

                    for item in analysis.action_items
                ],

                use_container_width=True,

                hide_index=True,
            )

        else:

            st.info(
                "No action items found."
            )

    # =========================================================================
    # DECISIONS
    # =========================================================================

    with tabs[2]:

        if analysis.key_decisions:

            for i, decision in enumerate(
                analysis.key_decisions,
                1,
            ):

                st.markdown(
                    f"**{i}.** {decision}"
                )

                st.divider()

        else:

            st.info(
                "No key decisions found."
            )

    # =========================================================================
    # QUESTIONS
    # =========================================================================

    with tabs[3]:

        if analysis.open_questions:

            for i, question in enumerate(
                analysis.open_questions,
                1,
            ):

                st.markdown(
                    f"**{i}.** {question}"
                )

        else:

            st.info(
                "No open questions found."
            )

    # =========================================================================
    # TOPICS
    # =========================================================================

    with tabs[4]:

        if analysis.key_topics:

            st.markdown(

                " ".join(
                    f'<span class="pill">{topic}</span>'
                    for topic in analysis.key_topics
                ),

                unsafe_allow_html=True,
            )

        else:

            st.info(
                "No topics extracted."
            )

    # =========================================================================
    # DOWNLOADS
    # =========================================================================

    with tabs[5]:

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(

                "⬇️ Download JSON Report",

                data=json.dumps(
                    analysis.to_json_dict(),
                    indent=2,
                    ensure_ascii=False,
                ),

                file_name=(
                    "meeting_analysis.json"
                ),

                mime="application/json",

                use_container_width=True,
            )

        with col2:

            st.download_button(

                "⬇️ Download Transcript (TXT)",

                data=analysis.transcript,

                file_name=(
                    "transcript.txt"
                ),

                mime="text/plain",

                use_container_width=True,
            )


# =============================================================================
# TRANSCRIPT PAGE
# =============================================================================

def render_transcript_page() -> None:

    analysis: Optional[
        MeetingAnalysis
    ] = st.session_state.analysis

    if analysis is None:

        st.info(
            "No transcript yet. Go to **Analyze** "
            "to process a video or recording."
        )

        return

    st.markdown(
        "#### Full Transcript"
    )

    st.text_area(
        "Transcript",

        analysis.transcript,

        height=420,

        label_visibility="collapsed",
    )

    st.caption(
        "Select the text above and copy with Ctrl/Cmd+C."
    )

    st.download_button(

        "⬇️ Download Transcript (TXT)",

        data=analysis.transcript,

        file_name="transcript.txt",

        mime="text/plain",
    )


# =============================================================================
# RAG SOURCE RENDERER
# =============================================================================

def render_sources(
    sources: list[dict],
) -> None:

    if not sources:

        return

    with st.expander(
        f"📚 Sources ({len(sources)})"
    ):

        for source in sources:

            chunk_id = source.get(
                "chunk_id"
            )

            score = source.get(
                "score"
            )

            text = source.get(
                "text",
                "",
            )

            st.markdown(
                f"**Chunk {chunk_id}**"
            )

            if score is not None:

                st.caption(
                    f"Similarity score: {score:.4f}"
                )

            st.write(
                text
            )

            st.divider()


# =============================================================================
# CHAT PAGE
# =============================================================================

def render_chat_page() -> None:

    """
    RAG chat flow:

        User question
              ↓
        HuggingFace query embedding
              ↓
        Qdrant Cloud retrieval
              ↓
        Top relevant chunks
              ↓
        Groq
              ↓
        Grounded answer
    """

    st.markdown(
        "#### 💬 Chat with Meeting"
    )

    # =========================================================================
    # NO MEETING
    # =========================================================================

    if (
        st.session_state.analysis is None
        or not st.session_state.meeting_id
        or not st.session_state.rag_indexed
    ):

        st.info(
            "Analyze a meeting first. "
            "Once the transcript is indexed, "
            "you can chat with it here."
        )

        return

    # =========================================================================
    # MEETING INFORMATION
    # =========================================================================

    analysis: MeetingAnalysis = (
        st.session_state.analysis
    )

    st.markdown(
        f"**Meeting:** {analysis.title}"
    )

    st.caption(
        f"Meeting ID: {st.session_state.meeting_id} "
        f"· {st.session_state.rag_chunk_count} "
        "semantic chunks indexed"
    )

    # =========================================================================
    # CLEAR CHAT
    # =========================================================================

    clear_col, info_col = st.columns(
        [1, 4]
    )

    with clear_col:

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):

            st.session_state.chat_messages = []

            st.rerun()

    st.divider()

    # =========================================================================
    # CHAT HISTORY
    # =========================================================================

    for message in (
        st.session_state.chat_messages
    ):

        role = message.get(
            "role",
            "assistant"
        )

        content = message.get(
            "content",
            ""
        )

        with st.chat_message(
            role
        ):

            st.markdown(
                content
            )

            # ---------------------------------------------------------------
            # Sources
            # ---------------------------------------------------------------

            if (
                role == "assistant"
                and message.get("sources")
            ):

                render_sources(
                    message["sources"]
                )

    # =========================================================================
    # USER QUESTION
    # =========================================================================

    question = st.chat_input(
        "Ask anything about this meeting..."
    )

    if not question:

        return

    question = question.strip()

    if not question:

        return

    # =========================================================================
    # DISPLAY USER QUESTION
    # =========================================================================

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # =========================================================================
    # GENERATE RAG ANSWER
    # =========================================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the meeting and generating an answer..."
        ):

            try:

                result = chat_with_meeting(

                    question=question,

                    meeting_id=(
                        st.session_state.meeting_id
                    ),

                    top_k=5,
                )

                answer = result.get(
                    "answer",
                    "I couldn't generate an answer."
                )

                sources = result.get(
                    "sources",
                    []
                )

                # -----------------------------------------------------------
                # Answer
                # -----------------------------------------------------------

                st.markdown(
                    answer
                )

                # -----------------------------------------------------------
                # Sources
                # -----------------------------------------------------------

                render_sources(
                    sources
                )

                # -----------------------------------------------------------
                # Save assistant response
                # -----------------------------------------------------------

                st.session_state.chat_messages.append(

                    {
                        "role":
                            "assistant",

                        "content":
                            answer,

                        "sources":
                            sources,
                    }
                )

            # =================================================================
            # BACKEND ERROR
            # =================================================================

            except BackendNotWiredError as e:

                error_message = (
                    f"RAG backend is not available: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.chat_messages.append(

                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )

            # =================================================================
            # VALUE ERROR
            # =================================================================

            except ValueError as e:

                error_message = str(e)

                st.warning(
                    error_message
                )

                st.session_state.chat_messages.append(

                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )

            # =================================================================
            # UNEXPECTED ERROR
            # =================================================================

            except Exception:

                print(
                    safe_traceback()
                )

                error_message = (
                    "I couldn't generate an answer right now. "
                    "Please check the Qdrant and Groq configuration."
                )

                st.error(
                    error_message
                )

                st.session_state.chat_messages.append(

                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )


# =============================================================================
# ROUTER
# =============================================================================

page = st.session_state.active_page


if page == "Analyze":

    render_analyze_page()


elif page == "Results":

    render_results_page()


elif page == "Transcript":

    render_transcript_page()


elif page == "Chat with Meeting":

    render_chat_page()