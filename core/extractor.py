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
        temperature=0.1
    )


def safe_json_parse(response: str, default: dict) -> dict:

    try:
        return json.loads(response)

    except json.JSONDecodeError:

        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1:

            try:
                return json.loads(
                    response[start:end + 1]
                )

            except json.JSONDecodeError:
                pass

        return default


def extract_meeting_information(transcript: str) -> dict:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting intelligence system.

Analyze the transcript and extract structured information.

Return ONLY valid JSON.

Use EXACTLY this structure:

{
    "action_items": [
        {
            "task": "",
            "owner": "",
            "deadline": "",
            "priority": "",
            "evidence": ""
        }
    ],

    "decisions": [
        {
            "decision": "",
            "made_by": "",
            "reason": "",
            "evidence": ""
        }
    ],

    "open_questions": [
        {
            "question": "",
            "context": "",
            "evidence": ""
        }
    ],

    "key_topics": [
        {
            "topic": "",
            "description": ""
        }
    ]
}

ACTION ITEMS:

Extract tasks that require an action.

Include:
- Explicit tasks
- Clearly implied tasks

Do NOT include:
- Completed historical tasks
- General statements
- Pure suggestions without actionable intent

If owner is unknown:
"Not specified"

If deadline is unknown:
"Not specified"

If priority is unknown:
"Not specified"


DECISIONS:

Extract decisions that were actually made.

Do not treat suggestions or discussions as decisions.

If none:
[]


OPEN QUESTIONS:

Extract questions or issues that remain unresolved.

Do not include questions that were already answered.

If none:
[]


KEY TOPICS:

Extract the major topics discussed.

Avoid duplicate topics.

IMPORTANT:

- Do not invent information.
- Preserve names and dates.
- Evidence must be based on the transcript.
- Return valid JSON only.
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
            "transcript": transcript
        }
    )

    return safe_json_parse(
        response,
        {
            "action_items": [],
            "decisions": [],
            "open_questions": [],
            "key_topics": []
        }
    )


# ---------------------------------------------------------
# Backward-compatible helper functions
# ---------------------------------------------------------

def extract_action_items(transcript: str) -> str:

    result = extract_meeting_information(transcript)

    action_items = result.get(
        "action_items",
        []
    )

    if not action_items:
        return "No action items found."

    output = []

    for index, item in enumerate(action_items, start=1):

        output.append(
            f"{index}. {item.get('task', 'Not specified')}\n"
            f"   Owner: {item.get('owner', 'Not specified')}\n"
            f"   Deadline: {item.get('deadline', 'Not specified')}\n"
            f"   Priority: {item.get('priority', 'Not specified')}"
        )

    return "\n".join(output)


def extract_key_decisions(transcript: str) -> str:

    result = extract_meeting_information(transcript)

    decisions = result.get(
        "decisions",
        []
    )

    if not decisions:
        return "No key decisions found."

    output = []

    for index, item in enumerate(decisions, start=1):

        output.append(
            f"{index}. {item.get('decision', 'Not specified')}\n"
            f"   Made by: {item.get('made_by', 'Not specified')}\n"
            f"   Reason: {item.get('reason', 'Not specified')}"
        )

    return "\n".join(output)


def extract_questions(transcript: str) -> str:

    result = extract_meeting_information(transcript)

    questions = result.get(
        "open_questions",
        []
    )

    if not questions:
        return "No open questions found."

    output = []

    for index, item in enumerate(questions, start=1):

        output.append(
            f"{index}. {item.get('question', 'Not specified')}\n"
            f"   Context: {item.get('context', 'Not specified')}"
        )

    return "\n".join(output)


def extract_key_topics(transcript: str) -> str:

    result = extract_meeting_information(transcript)

    topics = result.get(
        "key_topics",
        []
    )

    if not topics:
        return "No key topics found."

    output = []

    for index, item in enumerate(topics, start=1):

        output.append(
            f"{index}. {item.get('topic', 'Not specified')}: "
            f"{item.get('description', '')}"
        )

    return "\n".join(output)