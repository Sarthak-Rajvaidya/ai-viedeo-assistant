# core/classifier.py

import os
import json

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# ============================================================
# LLM CONFIGURATION
# ============================================================

def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in environment variables."
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.0
    )


# ============================================================
# CONTENT CLASSIFICATION
# ============================================================

def classify_content(transcript: str) -> dict:
    """
    Classifies a transcript into a high-level content type.

    Possible content types:
        - meeting
        - educational
        - interview
        - podcast
        - tutorial
        - presentation
        - discussion
        - lecture
        - other

    Returns:
        {
            "content_type": "...",
            "confidence": 0.0,
            "reason": "..."
        }
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert content classification system.

Your task is to analyze the provided transcript and classify
what type of content it represents.

Choose exactly ONE content type from:

- meeting
- educational
- interview
- podcast
- tutorial
- presentation
- discussion
- lecture
- other

Return ONLY valid JSON.

The JSON must follow this exact structure:

{{
    "content_type": "meeting",
    "confidence": 0.95,
    "reason": "Short explanation"
}}

Rules:

1. content_type must be exactly one of the allowed categories.
2. confidence must be a number between 0 and 1.
3. reason must be short and explain the classification.
4. Do not add markdown.
5. Do not add ```json.
6. Do not add any text outside the JSON.
"""
            ),
            (
                "human",
                """
Analyze this transcript:

{transcript}
"""
            )
        ]
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "transcript": transcript
        }
    )

    # --------------------------------------------------------
    # Clean response
    # --------------------------------------------------------

    response = response.strip()

    # Remove accidental markdown fences
    if response.startswith("```json"):
        response = response[7:]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:
        result = json.loads(response)

    except json.JSONDecodeError:
        print("Warning: Mistral returned invalid JSON.")
        print("Raw response:")
        print(response)

        # Safe fallback
        return {
            "content_type": "other",
            "confidence": 0.0,
            "reason": "Unable to parse classifier response."
        }

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    allowed_types = {
        "meeting",
        "educational",
        "interview",
        "podcast",
        "tutorial",
        "presentation",
        "discussion",
        "lecture",
        "other"
    }

    content_type = result.get("content_type", "other")

    if content_type not in allowed_types:
        content_type = "other"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = result.get(
        "reason",
        "No classification reason provided."
    )

    return {
        "content_type": content_type,
        "confidence": confidence,
        "reason": reason
    }


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_content_type(transcript: str) -> str:
    """
    Convenience function that returns only the content type.
    """

    result = classify_content(transcript)

    return result["content_type"]