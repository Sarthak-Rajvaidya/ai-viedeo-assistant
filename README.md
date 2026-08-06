# 🧠 AI Meeting Intelligence Assistant

An end-to-end AI-powered Meeting Assistant built using **Python**, **LLMs**, and **Retrieval-Augmented Generation (RAG)**.

The application can process meeting recordings from **YouTube URLs** or uploaded **audio/video files**, automatically generate meeting summaries, extract action items, identify key decisions, and allow users to chat with meeting transcripts using semantic search.

> 🚧 **Project Status:** Actively Under Development

---

# ✨ Features

## ✅ Implemented

- 🎥 Download audio from YouTube using **yt-dlp**
- 🎵 Automatic audio extraction & conversion to **WAV** using **FFmpeg**
- 📁 Automatic downloads directory creation
- 🔐 Secure API key management using `.env`
- ⚙️ Modular project setup for scalable development

---

## 🚀 Upcoming Features

- 🎙️ Audio & Video File Upload
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
- 🌐 Streamlit Web Interface

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
│
├── utils/
│   └── audio_processor.py
│
├── chroma_db/
├── reports/
├── uploads/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── app.py
```

> **Note:** The project structure will be refactored into a `src/` based architecture as development progresses.

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/AI-Meeting-Assistant-RAG.git
```

---

## 2. Navigate to Project

```bash
cd AI-Meeting-Assistant-RAG
```

---

## 3. Create Virtual Environment

```bash
py -3.11 -m venv venv
```

---

## 4. Activate Virtual Environment

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

Download FFmpeg and ensure both `ffmpeg` and `ffprobe` are available in your system PATH.

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
- Python Virtual Environment Setup
- Git & GitHub Setup
- README & .gitignore
- Mistral AI Integration
- FFmpeg Installation & Configuration
- YouTube Audio Downloader
- Automatic WAV Audio Conversion
- Modular Audio Processing Pipeline

---

## 🚧 Currently Working On

- Whisper Speech-to-Text Integration

---

# 📌 Development Roadmap

- [x] Project Setup
- [x] Environment Configuration
- [x] YouTube Audio Download
- [x] Audio Conversion using FFmpeg
- [ ] Whisper Transcription
- [ ] Meeting Summarization
- [ ] Action Item Extraction
- [ ] Decision Extraction
- [ ] Embedding Generation
- [ ] ChromaDB Integration
- [ ] RAG Chat
- [ ] Streamlit Dashboard
- [ ] PDF Report Generation
- [ ] Docker Deployment

---

# 📸 Demo

🚧 Screenshots and demo video will be added after the first working prototype.

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you'd like to contribute, feel free to fork the repository and submit a pull request.

---

# 📄 License

This project is licensed under the MIT License.

---

## ⭐ If you like this project, consider giving it a star!