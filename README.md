# 🧠 AI Meeting Intelligence Assistant

An end-to-end AI-powered Meeting Assistant built using Python and Retrieval-Augmented Generation (RAG).

The application automatically transcribes meetings, generates intelligent summaries, extracts action items, and allows users to chat with meeting transcripts using a vector database.

> 🚧 Project Status: In Development

---

## 🚀 Planned Features

- 🎙️ Audio & Video Upload
- 🔗 YouTube URL Support
- 📝 Speech-to-Text using Whisper
- 🌍 Hindi/Hinglish Transcription using Sarvam AI
- 📄 AI Meeting Summarization
- ✅ Action Item Extraction
- 📌 Key Decision Extraction
- ❓ Open Questions Detection
- 💬 Chat with Meeting (RAG)
- 🧠 Semantic Search using ChromaDB
- 📑 PDF & TXT Report Export

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Mistral AI
- OpenAI Whisper
- Sarvam AI
- LangChain
- HuggingFace Embeddings
- ChromaDB
- python-dotenv

---

## 📂 Project Structure

```
AI-Meeting-Assistant/
│
├── app.py
├── .env
├── requirements.txt
├── README.md
├── .gitignore
│
├── uploads/
├── chroma_db/
├── reports/
│
├── src/
│   ├── transcription/
│   ├── summarizer/
│   ├── rag/
│   ├── embeddings/
│   ├── prompts/
│   └── utils/
│
└── assets/
```

---

## ⚙️ Setup

### Clone Repository

```bash
git clone https://github.com/your-username/AI-Meeting-Assistant.git
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## 📌 Current Progress

- ✅ Project Initialized
- ✅ Virtual Environment Created
- ✅ Mistral API Key Configured
- ⏳ Development Started

---

## 📜 License

This project is licensed under the MIT License.