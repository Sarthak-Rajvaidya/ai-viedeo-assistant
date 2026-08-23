import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in .env")

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting summarizer.

Summarize this portion of a transcript concisely.

Focus on:
- Important topics
- Main ideas
- Important facts
- Decisions
- Tasks
- Conclusions

Do not invent information.
"""
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for chunk in chunks:

        summary = map_chain.invoke(
            {
                "text": chunk
            }
        )

        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting summarizer.

Combine the partial summaries into one professional summary.

Use the following structure:

### Overview
Short overview.

### Key Discussion Points
- Important point
- Important point

### Main Takeaways
- Takeaway
- Takeaway

### Conclusions
- Conclusion

Rules:
- Do not invent information.
- Remove duplicate information.
- Keep the summary concise.
- Preserve important names, dates and numbers.
"""
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke(
        {
            "text": combined
        }
    ).strip()


def generate_title(transcript: str) -> str:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Based on the transcript, generate a short professional title.

Rules:

- Maximum 8 words.
- Clearly represent the main topic.
- Do not use quotation marks.
- Return ONLY the title.
"""
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke(
        {
            "text": transcript[:4000]
        }
    ).strip()