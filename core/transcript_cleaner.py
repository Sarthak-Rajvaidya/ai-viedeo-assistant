import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set in .env")

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.0
    )


def clean_transcript(transcript: str) -> str:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a professional transcript editor.

Clean the provided speech-to-text transcript while preserving
the original meaning and information.

Rules:

1. Fix obvious grammar mistakes.
2. Fix incorrect spacing and punctuation.
3. Fix obvious speech-to-text artifacts.
4. Remove accidental repeated words or phrases.
5. Preserve names, numbers, dates, technical terms and important details.
6. Do not add information that is not present.
7. Do not summarize.
8. Do not change the meaning.
9. Do not remove important content.
10. Return ONLY the cleaned transcript.
"""
            ),
            (
                "human",
                "{transcript}"
            )
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "transcript": transcript
        }
    ).strip()