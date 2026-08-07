# 🧠 AI Meeting Intelligence Assistant

An end-to-end AI-powered Meeting Assistant built using **Python**, **Large Language Models (LLMs)**, and **Retrieval-Augmented Generation (RAG)**.

The application processes meeting recordings from **YouTube URLs** or uploaded **audio/video files**, automatically transcribes conversations, generates intelligent meeting summaries, extracts action items, identifies key decisions, and allows users to chat with meeting transcripts using semantic search.

> 🚧 **Project Status:** Actively Under Development

---

# ✨ Features

## ✅ Implemented

### 🎥 Audio Acquisition

- Download meeting audio directly from **YouTube URLs**
- Support for local **audio & video files**
- Automatic downloads directory creation

### 🎵 Audio Preprocessing

- Convert any supported media format to **WAV**
- Normalize audio to **Mono (1 Channel)**
- Resample audio to **16 kHz** (Whisper-compatible)
- Automatically split long recordings into configurable chunks
- Prepare audio for efficient speech-to-text transcription

### ⚙️ Project Setup

- Secure API key management using `.env`
- FFmpeg integration
- Modular Python project structure
- Python virtual environment setup

---

# 🚀 Upcoming Features

- 📝 Speech-to-Text using OpenAI Whisper
- 🌍 Hindi & Hinglish Transcription using Sarvam AI
- 📄 AI Meeting Summarization
- ✅ Action Item Extraction
- 📌 Key Decision Extraction
- ❓ Open Question Detection
- 💬 Chat with Meeting using RAG
- 🧠 Semantic Search using ChromaDB
- 🔍 HuggingFace Embeddings
- 📑 Export Meeting Report as PDF & TXT
- 🌐 Interactive Streamlit Dashboard

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.11 |
| UI | Streamlit |
| LLM | Mistral AI |
| Speech-to-Text | OpenAI Whisper, Sarvam AI |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| Audio Processing | yt-dlp, FFmpeg, PyDub |
| Environment | python-dotenv |

---

# 📂 Project Structure

```text
AI-Meeting-Assistant-RAG/
│
├── downloads/
├── uploads/
├── reports/
├── chroma_db/
│
├── utils/
│   └── audio_processor.py
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── app.py
```

> 📌 The project will be migrated to a modular `src/` architecture as additional AI components are implemented.

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/AI-Meeting-Assistant-RAG.git
```

---

## 2. Navigate into Project

```bash
cd AI-Meeting-Assistant-RAG
```

---

## 3. Create Virtual Environment

```bash
py -3.11 -m venv venv
```

---

## 4. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Install FFmpeg

Install FFmpeg and ensure both `ffmpeg` and `ffprobe` are available in your system PATH.

Verify installation:

```bash
ffmpeg -version
```

```bash
ffprobe -version
```

---

## 7. Configure Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

# 🚀 Current Progress

## ✅ Completed

- Project Initialization
- Git Repository Setup
- Python Virtual Environment
- README & .gitignore
- Mistral AI Configuration
- FFmpeg Installation & Configuration
- YouTube Audio Downloader
- Local Audio & Video File Support
- Audio Format Conversion to WAV
- Audio Normalization (Mono + 16 kHz)
- Audio Chunking for Long Meetings
- Modular Audio Processing Pipeline

---

## 🚧 Currently Working On

- Whisper Speech-to-Text Integration

---

# 📌 Development Roadmap

- [x] Project Setup
- [x] Environment Configuration
- [x] YouTube Audio Downloader
- [x] Local Audio & Video Upload Support
- [x] Audio Conversion to WAV
- [x] Audio Normalization
- [x] Audio Chunking
- [ ] Whisper Speech-to-Text
- [ ] AI Meeting Summarization
- [ ] Action Item Extraction
- [ ] Decision Extraction
- [ ] Open Question Detection
- [ ] Embedding Generation
- [ ] ChromaDB Integration
- [ ] Conversational RAG
- [ ] Streamlit Dashboard
- [ ] PDF & TXT Report Export
- [ ] Docker Deployment

---

# 🔄 Processing Pipeline

```text
                User Input
                     │
         ┌───────────┴────────────┐
         │                        │
   YouTube URL             Audio / Video File
         │                        │
      yt-dlp                PyDub + FFmpeg
         │                        │
         └──────────┬─────────────┘
                    │
          WAV Audio (Mono • 16 kHz)
                    │
             Audio Chunking
                    │
             Whisper (Next)
                    │
              Meeting Transcript
                    │
                 Mistral AI
                    │
        Summary • Decisions • Actions
                    │
      HuggingFace Embeddings
                    │
               ChromaDB
                    │
             Chat with Meeting
```

---

# 📸 Demo

🚧 Screenshots and demonstration video will be added after the first working prototype.

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork this repository and submit a Pull Request.

---

# 📄 License

Licensed under the MIT License.

---

⭐ **If you found this project helpful, consider giving it a Star!**