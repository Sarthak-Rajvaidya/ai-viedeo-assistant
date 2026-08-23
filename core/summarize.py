from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


# ============================================================
# LLM CONFIGURATION
# ============================================================

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


# ============================================================
# TRANSCRIPT CHUNKING
# ============================================================

def split_transcript(transcript: str) -> list:
    """
    Split a long transcript into smaller overlapping chunks
    so that the LLM can process it safely.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# ============================================================
# MEETING SUMMARIZATION
# ============================================================

def summarize(transcript: str) -> str:

    llm = get_llm()

    # --------------------------------------------------------
    # STEP 1: Summarize individual transcript chunks
    # --------------------------------------------------------

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Summarize this portion of a meeting transcript "
                "concisely while preserving important information, "
                "decisions, tasks, and discussion points."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({
            "text": chunk
        })

        chunk_summaries.append(summary)

    # --------------------------------------------------------
    # STEP 2: Combine all partial summaries
    # --------------------------------------------------------

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Combine the following partial meeting summaries "
                "into one professional and coherent meeting summary. "
                "Use clear bullet points. "
                "Preserve important topics, decisions, tasks, "
                "and conclusions. "
                "Do not add information that is not present "
                "in the provided summaries."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    # Simple and correct LangChain pipeline
    combined_chain = (
        combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke({
        "text": combined
    })


# ============================================================
# MEETING TITLE GENERATION
# ============================================================

def generate_title(transcript: str) -> str:

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a "
                "short professional meeting title. "
                "Maximum 8 words. "
                "Return only the title and nothing else."
            ),
            (
                "human",
                "{text}"
            ),
        ]
    )

    title_chain = (
        title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke({
        "text": transcript[:2000]
    })