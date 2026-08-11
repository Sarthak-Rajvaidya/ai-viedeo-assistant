# 🧠 AI Meeting Intelligence Assistant

An end-to-end AI-powered Meeting Intelligence Assistant built using **Python, Large Language Models (LLMs), Speech-to-Text, Vector Databases, and Retrieval-Augmented Generation (RAG)**.

The application processes meeting recordings from **YouTube URLs or local audio/video files**, converts speech into text, supports English and Hindi/Hinglish meetings, and prepares transcripts for downstream AI-powered meeting analysis.

The planned system will generate summaries, extract action items, identify key decisions and open questions, and allow users to chat with their meetings using RAG.

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

### 📝 English Speech-to-Text

- Local transcription using **OpenAI Whisper**
- Whisper model loaded locally
- Chunk-based transcription for long recordings
- Support for transcription and translation tasks
- Full transcript generation by combining chunk results

### 🌍 Hindi Speech Processing

- Hindi speech recognition using **Sarvam AI Saaras v3**
- Sarvam Batch Speech-to-Text API for long recordings
- Hindi audio → Hindi transcript
- Hindi audio → **English translation**
- Support for speaker diarization
- JSON output parsing and transcript extraction

### ⚙️ Project Infrastructure

- Secure API key management using `.env`
- FFmpeg integration
- Python 3.11 virtual environment
- Modular project architecture
- Separate audio processing and transcription modules
- Git/GitHub version control

---

# 🚀 Upcoming Features

### 🤖 Meeting Intelligence

- 📄 AI Meeting Summarization
- ✅ Action Item Extraction
- 👤 Action Item Owner Detection
- 📅 Deadline Detection
- 📌 Key Decision Extraction
- ❓ Open Question Detection
- 🔎 Important Topic Extraction

### 🧠 RAG Pipeline

- Text Chunking
- HuggingFace Embedding Generation
- Semantic Search
- ChromaDB Vector Storage
- Retrieval-Augmented Generation
- Conversational Chat with Meeting
- Context-aware question answering

### 🤖 LLM Integration

- Mistral AI powered meeting analysis
- Structured outputs for meeting information
- Prompt engineering
- Multi-step LLM pipelines

### 🌐 Application Layer

- Streamlit Web Interface
- YouTube URL input
- Audio/video file upload
- Meeting processing dashboard
- Transcript viewer
- AI-generated meeting report
- Interactive chat interface

### 📑 Export & Deployment

- PDF Meeting Report
- TXT Transcript Export
- Docker support
- Production deployment

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

The project uses a modular pipeline where different components are responsible for different stages of the meeting intelligence workflow.

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