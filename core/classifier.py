import os
import json

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


def classify_content(transcript: str) -> dict:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a content classification system.

Classify the provided transcript into exactly ONE category:

- meeting
- project_meeting
- standup
- interview
- lecture
- tutorial
- webinar
- brainstorming
- presentation
- conversation
- other

Return ONLY valid JSON.

Required format:

{
    "content_type": "project_meeting",
    "confidence": 0.95,
    "reason": "Short explanation"
}

Rules:

- confidence must be between 0 and 1.
- Do not invent information.
- Choose the category that best represents the transcript.
"""
            ),
            (
                "human",
                "{transcript}"
            )
        ]
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "transcript": transcript[:12000]
        }
    )

    try:
        return json.loads(response)

    except json.JSONDecodeError:

        # Try to recover JSON if model added extra text.
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {
            "content_type": "other",
            "confidence": 0.0,
            "reason": "Unable to parse classifier response."
        }