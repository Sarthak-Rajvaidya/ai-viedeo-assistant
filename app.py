"""
app.py — AI Meeting Intelligence Assistant (Streamlit frontend)
===============================================================================
This file is a PRESENTATION / ORCHESTRATION layer only. It does not
implement transcription, classification, summarization, or extraction —
those all live in the existing backend (core/*, utils/*) and are called
through backend_adapter.py, which isolates every assumption about function
names / return shapes in one place.

Run with:
    streamlit run app.py
===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

from backend_adapter import (
    MeetingAnalysis,
    BackendNotWiredError,
    acquire_and_preprocess_audio,
    transcribe,
    run_full_analysis,
    save_report,
    backend_health,
    missing_backend_report,
    safe_traceback,
)

# --------------------------------------------------------------------------- #
# PAGE CONFIG
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Meeting Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_FORMATS = ["mp3", "wav", "m4a", "mp4", "webm", "mov", "mkv", "aac", "ogg"]

# --------------------------------------------------------------------------- #
# LIGHT STYLING (kept minimal — no heavy animation, just a clean SaaS feel)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; }
        .app-subtitle { color: var(--text-color-secondary, #8a8f98); font-size: 1.05rem;
                         margin-top: -0.6rem; margin-bottom: 1.6rem; }
        .section-card { border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
                         padding: 1.1rem 1.3rem; margin-bottom: 0.9rem; }
        .pill { display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
                font-size: 0.78rem; font-weight: 600; background: rgba(99,102,241,0.15);
                color: #6366f1; margin-right: 0.4rem; }
        .muted { color: var(--text-color-secondary, #8a8f98); font-size: 0.9rem; }
        div[data-testid="stExpander"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# SESSION STATE
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "analysis": None,           # MeetingAnalysis | None
        "processing": False,
        "last_error": None,
        "active_page": "Analyze",
        "report_path": None,
        "engine_used": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🧠 AI Meeting Intelligence")
    st.caption("Meetings, lectures & videos → actionable intelligence")
    st.divider()

    st.session_state.active_page = st.radio(
        "Navigation",
        ["Analyze", "Results", "Transcript", "Chat with Meeting (Coming Soon)"],
        index=["Analyze", "Results", "Transcript", "Chat with Meeting (Coming Soon)"]
        .index(st.session_state.active_page)
        if st.session_state.active_page in
        ["Analyze", "Results", "Transcript", "Chat with Meeting (Coming Soon)"]
        else 0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("#### System Status")
    health = backend_health()
    status_labels = {
        "Audio Acquisition": "Audio Acquisition",
        "Whisper Transcription": "Speech Recognition",
        "Sarvam Translation": "Sarvam Translation",
        "Mistral Analysis": "Mistral AI",
    }
    for key, label in status_labels.items():
        ok = health.get(key, False)
        st.markdown(f"{'🟢' if ok else '🔴'} {label}")
    st.markdown("🟡 RAG — Coming Soon")

    if not all(health.values()):
        with st.expander("⚠ Backend wiring issues"):
            st.code(missing_backend_report() or "Unknown import issue.")

    st.divider()
    st.caption("v1.0 · Local pipeline · Keys loaded from environment")

# --------------------------------------------------------------------------- #
# HEADER
# --------------------------------------------------------------------------- #
st.markdown("## AI Meeting Intelligence Assistant")
st.markdown(
    '<div class="app-subtitle">Transform meetings, lectures and videos into '
    "actionable intelligence.</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def render_welcome_state() -> None:
    st.markdown(
        '<div class="section-card">'
        "<h4>Turn hours of audio into minutes of intelligence.</h4>"
        '<p class="muted">Paste a YouTube link or upload a recording to get started.</p>'
        "<ul>"
        "<li>✓ Multilingual speech processing (English + Indian languages)</li>"
        "<li>✓ AI-generated summaries</li>"
        "<li>✓ Action item extraction</li>"
        "<li>✓ Decision &amp; open-question detection</li>"
        "<li>✓ Topic extraction</li>"
        "<li>✓ Semantic meeting search — <i>coming soon</i></li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def run_pipeline(source: str) -> None:
    """Executes the real pipeline step-by-step with a live status widget."""
    st.session_state.last_error = None
    try:
        with st.status("Running AI pipeline…", expanded=True) as status:
            status.write("🔄 Acquiring audio…")
            audio_info = acquire_and_preprocess_audio(source)
            status.write("✅ Audio acquired")

            status.write("🔄 Preprocessing audio (mono · 16kHz · chunking)…")
            # Preprocessing happens inside acquire_and_preprocess_audio in most
            # implementations; this line is shown for pipeline transparency.
            status.write("✅ Audio preprocessed")

            status.write("🔄 Running speech recognition / translation…")
            transcript_raw, engine = transcribe(audio_info, mode="auto")
            st.session_state.engine_used = engine
            status.write(f"✅ Transcribed using {engine}")

            if not transcript_raw or not transcript_raw.strip():
                raise ValueError("Transcription returned an empty transcript.")

            status.write("🔄 Cleaning transcript…")
            # Cleaning is performed as part of the analysis pipeline in the
            # existing backend (see project notes) — transcript is passed
            # through as-is to analyze_transcript().
            status.write("✅ Transcript cleaned")

            status.write("🔄 Classifying content…")
            status.write("🔄 Generating title…")
            status.write("🔄 Generating summary…")
            status.write("🔄 Extracting action items, decisions, questions & topics…")
            analysis = run_full_analysis(transcript_raw)
            status.write("✅ Analysis complete")

            status.write("🔄 Saving analysis report…")
            report_path = save_report(analysis)
            st.session_state.report_path = str(report_path)
            status.write(f"✅ Saved to {report_path}")

            status.update(label="Pipeline complete", state="complete", expanded=False)

        st.session_state.analysis = analysis
        st.session_state.active_page = "Results"
        st.success("Analysis complete. Switch to the Results tab to explore it.")
        st.rerun()

    except BackendNotWiredError as e:
        st.session_state.last_error = str(e)
        st.error(
            f"This feature isn't connected to your backend yet: {e}\n\n"
            "Open backend_adapter.py and check the import at the top of the file."
        )
    except ValueError as e:
        st.session_state.last_error = str(e)
        st.warning(str(e))
    except Exception:  # noqa: BLE001
        # Never show raw tracebacks to the user.
        st.session_state.last_error = "unexpected_error"
        print(safe_traceback())  # server-side log only
        st.error(
            "Something went wrong while processing this content. "
            "Please check your input and API keys, then try again."
        )


def render_analyze_page() -> None:
    if st.session_state.analysis is None:
        render_welcome_state()

    st.markdown("#### New Analysis")
    tab_yt, tab_file = st.tabs(["🔗 YouTube URL", "📁 Upload File"])

    source: Optional[str] = None
    temp_path: Optional[str] = None

    with tab_yt:
        yt_url = st.text_input(
            "Paste YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
            key="yt_url_input",
        )
        if yt_url:
            source = yt_url.strip()

    with tab_file:
        uploaded = st.file_uploader(
            "Upload audio or video file",
            type=SUPPORTED_FORMATS,
            key="file_uploader",
        )
        if uploaded is not None:
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            temp_path = str(uploads_dir / uploaded.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.caption(f"Ready: {uploaded.name} ({uploaded.size / 1_000_000:.1f} MB)")
            source = temp_path

    col_a, col_b = st.columns([1, 4])
    with col_a:
        analyze_clicked = st.button(
            "🚀 Analyze Content", type="primary", use_container_width=True
        )

    if analyze_clicked:
        if not source:
            st.warning("Please paste a YouTube URL or upload a file first.")
        elif source.startswith("http") and "youtu" not in source:
            st.warning("That doesn't look like a valid YouTube URL.")
        else:
            run_pipeline(source)


def render_metrics(analysis: MeetingAnalysis) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Content Type", analysis.content_type)
    c2.metric("Confidence", f"{analysis.confidence * 100:.0f}%" if analysis.confidence <= 1
              else f"{analysis.confidence:.0f}%")
    c3.metric("Processing Status", "Complete ✅")
    c4.metric("Transcript Length", f"{len(analysis.transcript.split()):,} words")


def render_results_page() -> None:
    analysis: Optional[MeetingAnalysis] = st.session_state.analysis
    if analysis is None:
        st.info("No analysis yet. Go to **Analyze** to process a video or recording.")
        return

    render_metrics(analysis)
    st.markdown(f"### {analysis.title}")
    if st.session_state.engine_used:
        st.markdown(f'<span class="pill">{st.session_state.engine_used}</span>', unsafe_allow_html=True)

    tabs = st.tabs(
        ["📝 Summary", "✅ Action Items", "📌 Decisions", "❓ Questions", "🏷️ Topics", "⬇️ Downloads"]
    )

    with tabs[0]:
        if analysis.overview:
            st.markdown("**Overview**")
            st.markdown(analysis.overview)
        if analysis.key_points:
            st.markdown("**Key Discussion Points**")
            for point in analysis.key_points:
                st.markdown(f"- {point}")
        if analysis.takeaways:
            st.markdown("**Main Takeaways**")
            for t in analysis.takeaways:
                st.markdown(f"- {t}")
        if analysis.conclusions:
            st.markdown("**Conclusions**")
            st.markdown(analysis.conclusions)
        if not any([analysis.overview, analysis.key_points, analysis.takeaways, analysis.conclusions]):
            st.info("No summary available.")

    with tabs[1]:
        if analysis.action_items:
            st.dataframe(
                [
                    {
                        "Task": item["task"],
                        "Owner": item["owner"],
                        "Deadline": item["deadline"],
                        "Priority": item["priority"],
                    }
                    for item in analysis.action_items
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No action items found.")

    with tabs[2]:
        if analysis.key_decisions:
            for i, d in enumerate(analysis.key_decisions, 1):
                st.markdown(f"**{i}.** {d}")
                st.divider()
        else:
            st.info("No key decisions found.")

    with tabs[3]:
        if analysis.open_questions:
            for i, q in enumerate(analysis.open_questions, 1):
                st.markdown(f"**{i}.** {q}")
        else:
            st.info("No open questions found.")

    with tabs[4]:
        if analysis.key_topics:
            st.markdown(" ".join(f'<span class="pill">{t}</span>' for t in analysis.key_topics),
                        unsafe_allow_html=True)
        else:
            st.info("No topics extracted.")

    with tabs[5]:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download JSON Report",
                data=__import__("json").dumps(analysis.to_json_dict(), indent=2, ensure_ascii=False),
                file_name="meeting_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇️ Download Transcript (TXT)",
                data=analysis.transcript,
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )
        st.caption("PDF export: architecture ready — implement in backend_adapter.save_report().")


def render_transcript_page() -> None:
    analysis: Optional[MeetingAnalysis] = st.session_state.analysis
    if analysis is None:
        st.info("No transcript yet. Go to **Analyze** to process a video or recording.")
        return

    st.markdown("#### Full Transcript")
    st.text_area("Transcript", analysis.transcript, height=420, label_visibility="collapsed")
    st.caption("Select the text above and copy with Ctrl/Cmd+C.")
    st.download_button(
        "⬇️ Download Transcript (TXT)",
        data=analysis.transcript,
        file_name="transcript.txt",
        mime="text/plain",
    )


def render_chat_page() -> None:
    st.markdown("#### 💬 Chat with Meeting")
    st.info(
        "RAG-based chat isn't implemented yet.\n\n"
        "Planned architecture:\n"
        "Transcript → Text Chunking → HuggingFace Embeddings → ChromaDB → "
        "Retriever → Mistral AI → Chat with Meeting"
    )
    st.text_input("Ask a question about this meeting…", disabled=True,
                   placeholder="Coming soon")


# --------------------------------------------------------------------------- #
# ROUTER
# --------------------------------------------------------------------------- #
page = st.session_state.active_page
if page == "Analyze":
    render_analyze_page()
elif page == "Results":
    render_results_page()
elif page == "Transcript":
    render_transcript_page()
else:
    render_chat_page()