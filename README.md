# 🧠 AI Meeting Intelligence Assistant

An end-to-end AI-powered Meeting Intelligence Assistant built using **Python, Large Language Models (LLMs), Speech-to-Text, Vector Databases, and Retrieval-Augmented Generation (RAG)**.

The application processes meeting recordings from **YouTube URLs or local audio/video files**, converts speech into text, supports multilingual meeting transcription, and performs AI-powered meeting analysis.

The system is being developed as a complete pipeline for **transcription → meeting intelligence → semantic search → conversational RAG**.

> 🚧 **Project Status:** Actively Under Development

---

# ✨ Features

## ✅ Implemented

### 🎥 Audio Acquisition

- Download meeting audio directly from **YouTube URLs**
- Support for local **audio and video files**
- Automatic download directory creation
- Unified input processing pipeline

### 🎵 Audio Preprocessing

- Convert supported audio/video formats to **WAV**
- Normalize audio to **Mono (1 Channel)**
- Resample audio to **16 kHz**
- Split long recordings into configurable audio chunks
- Prepare audio for efficient speech-to-text processing
- FFmpeg-powered audio conversion

### 📝 English Speech-to-Text

- Local transcription using **OpenAI Whisper**
- Local Whisper model loading
- Chunk-based transcription for long recordings
- Support for transcription and translation tasks
- Full transcript generation by combining chunk results

### 🌍 Hindi Speech Processing

- Hindi speech recognition using **Sarvam AI Saaras v3**
- Sarvam Batch Speech-to-Text API for long recordings
- Hindi audio → Hindi transcript
- Hindi audio → **English transcript**
- Speaker diarization support
- JSON output parsing
- Full transcript generation from batch results

### 🤖 AI Meeting Summarization

- Meeting transcript summarization using **Mistral AI**
- Hierarchical / map-reduce style summarization
- Transcript chunking before LLM processing
- Individual chunk summaries
- Final combined meeting summary
- Professional meeting summary generation
- Automatic meeting title generation

### 📌 Meeting Action Item Extraction

- Extract actionable tasks from meeting transcripts
- Identify task descriptions
- Detect responsible owners when explicitly mentioned
- Detect deadlines when mentioned
- Prevent unsupported task/owner/deadline generation
- Handles meetings with no identifiable action items

### ⚙️ Project Infrastructure

- Secure API key management using `.env`
- FFmpeg integration
- Python 3.11 virtual environment
- Modular project architecture
- Separate audio processing, transcription, and AI analysis modules
- Git/GitHub version control

---

# 🚀 Upcoming Features

## 🧠 Meeting Intelligence

- 📌 Key Decision Extraction
- ❓ Open Question Detection
- 🔎 Important Topic Extraction
- 👤 Advanced Speaker-aware Analysis
- 📅 Improved Deadline Detection
- 📊 Meeting Insights & Analytics
- 🛡️ Evidence-grounded extraction

---

## 🧠 RAG Pipeline

- Text Chunking for Retrieval
- HuggingFace Embedding Generation
- Semantic Search
- ChromaDB Vector Storage
- Retrieval-Augmented Generation
- Conversational Chat with Meeting
- Context-aware Question Answering
- Source/evidence-based responses

---

## 🤖 Advanced LLM Features

- Structured JSON outputs
- Pydantic-based structured responses
- Advanced prompt engineering
- Multi-step LLM pipelines
- Hallucination reduction
- Grounded meeting insights
- Context-aware analysis
- Meeting-specific AI agents/workflows

---

## 🌐 Application Layer

- Streamlit Web Interface
- YouTube URL input
- Audio/video file upload
- Meeting processing dashboard
- Transcript viewer
- AI-generated meeting report
- Meeting analytics dashboard
- Interactive chat interface

---

## 📑 Export & Deployment

- PDF Meeting Report
- TXT Transcript Export
- Docker support
- Production deployment
- API-based backend
- Cloud deployment

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit |
| LLM | Mistral AI |
| Speech-to-Text | OpenAI Whisper, Sarvam AI Saaras v3 |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| Audio Processing | PyDub, FFmpeg |
| YouTube Processing | yt-dlp |
| Environment Management | python-dotenv |
| PDF Generation | ReportLab |
| Version Control | Git, GitHub |

---

# 🏗️ System Architecture

The project follows a modular pipeline where each component handles a specific stage of the meeting intelligence workflow.

```text
                         USER INPUT
                             │
               ┌─────────────┴─────────────┐
               │                           │
         YouTube URL                Local Audio/Video
               │                           │
            yt-dlp                   PyDub + FFmpeg
               │                           │
               └─────────────┬─────────────┘
                             │
                             ▼
                    Audio Preprocessing
                             │
                     Mono + 16 kHz WAV
                             │
                             ▼
                      Audio Chunking
                             │
               ┌─────────────┴─────────────┐
               │                           │
         English Audio                Hindi Audio
               │                           │
               ▼                           ▼
        OpenAI Whisper              Sarvam Saaras v3
               │                           │
               │                    ┌──────┴──────┐
               │                    │             │
               │                 Hindi Text   English Text
               │
               └─────────────┬─────────────┘
                             │
                             ▼
                    Unified Transcript
                             │
                             ▼
                         Mistral AI
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
           Summary       Decisions      Action Items
              │              │              │
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                      Text Chunking
                             │
                             ▼
                  HuggingFace Embeddings
                             │
                             ▼
                         ChromaDB
                             │
                             ▼
                           RAG
                             │
                             ▼
                  Chat with Your Meeting




AI-viedeo-assistant-RAG/
│
├── core/
│   ├── transcriber.py
│   ├── sarvam_transcriber.py
│   ├── summarizer.py
│   └── extractor.py
│
├── utils/
│   └── audio_processor.py
│
├── downloads/
├── uploads/
├── reports/
├── chroma_db/
├── sarvam_output/
│
├── test.py
├── test_sarvam.py
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── app.py