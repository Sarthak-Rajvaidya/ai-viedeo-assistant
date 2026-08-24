import os
import json

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================
# LLM
# ============================================================

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.1
    )


# ============================================================
# LLM CHAIN
# ============================================================

def build_chain(system_prompt: str):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{transcript}")
        ]
    )

    return prompt | llm | StrOutputParser()


# ============================================================
# ACTION ITEMS
# ============================================================

def extract_action_items(transcript: str) -> str:

    chain = build_chain(
        """
You are an expert meeting intelligence analyst.

Analyze the transcript and identify ALL actionable tasks.

For every action item identify:

- Task
- Owner
- Deadline

Rules:

1. Only extract actions that are actually supported by the transcript.
2. Do not invent owners.
3. Do not invent deadlines.
4. If owner is unknown, write "Not specified".
5. If deadline is unknown, write "Not specified".
6. Return a numbered list.
7. If there are no action items, return exactly:
   "No action items found."

Example format:

1. Task: Prepare the project report
   Owner: Rahul
   Deadline: Friday

2. Task: Review the API implementation
   Owner: Not specified
   Deadline: Not specified
"""
    )

    return chain.invoke({"transcript": transcript})


# ============================================================
# KEY DECISIONS
# ============================================================

def extract_key_decisions(transcript: str) -> str:

    chain = build_chain(
        """
You are an expert meeting intelligence analyst.

Analyze the transcript and identify all important decisions that were
actually made during the discussion.

Rules:

1. Extract only decisions explicitly supported by the transcript.
2. Do not confuse suggestions with final decisions.
3. Do not invent information.
4. Return a numbered list.
5. If no decisions were made, return exactly:
   "No key decisions found."

Example:

1. The team decided to use PostgreSQL as the primary database.

2. The deployment will be moved to AWS next month.
"""
    )

    return chain.invoke({"transcript": transcript})


# ============================================================
# OPEN QUESTIONS
# ============================================================

def extract_questions(transcript: str) -> str:

    chain = build_chain(
        """
You are an expert meeting intelligence analyst.

Analyze the transcript and identify unresolved questions,
uncertainties, or topics that require follow-up.

Rules:

1. Extract only unresolved questions.
2. Do not convert answered questions into open questions.
3. Do not invent questions.
4. Return a numbered list.
5. If no unresolved questions exist, return exactly:
   "No open questions found."

Example:

1. Which database will be used for production?

2. Who will be responsible for deployment?
"""
    )

    return chain.invoke({"transcript": transcript})


# ============================================================
# IMPORTANT TOPICS
# ============================================================

def extract_topics(transcript: str) -> str:

    chain = build_chain(
        """
You are an expert meeting intelligence analyst.

Identify the most important topics discussed in the transcript.

Rules:

1. Extract meaningful topics rather than individual sentences.
2. Avoid unnecessary repetition.
3. Return a numbered list.
4. Keep each topic concise.
5. If no meaningful topics can be identified, return:
   "No important topics found."

Example:

1. English learning strategy
2. Daily vocabulary practice
3. Grammar improvement
4. Speaking practice using ChatGPT
"""
    )

    return chain.invoke({"transcript": transcript})


# ============================================================
# RISKS / BLOCKERS
# ============================================================

def extract_risks_and_blockers(transcript: str) -> str:

    chain = build_chain(
        """
You are an expert meeting intelligence analyst.

Identify any risks, blockers, problems, constraints, or challenges
mentioned in the transcript.

Rules:

1. Extract only issues supported by the transcript.
2. Do not invent risks.
3. Return a numbered list.
4. If no risks or blockers are mentioned, return exactly:
   "No risks or blockers found."

Example:

1. Limited spoken English confidence is a current challenge.

2. Lack of consistent daily practice may slow progress.
"""
    )

    return chain.invoke({"transcript": transcript})


# ============================================================
# STRUCTURED EXTRACTION
# ============================================================

def extract_meeting_information(transcript: str) -> dict:

    llm = get_llm()

    system_prompt = """
You are an expert meeting intelligence analyst.

Analyze the following transcript and extract structured information.

Return ONLY valid JSON.

Do not include markdown.
Do not include ```json.
Do not include explanations outside the JSON.

Use exactly this structure:

{{
    "action_items": [
        {{
            "task": "string",
            "owner": "string",
            "deadline": "string"
        }}
    ],
    "key_decisions": [
        "string"
    ],
    "open_questions": [
        "string"
    ],
    "important_topics": [
        "string"
    ],
    "risks_and_blockers": [
        "string"
    ]
}}

Rules:

- Extract only information supported by the transcript.
- Never invent facts.
- If owner is unknown, use "Not specified".
- If deadline is unknown, use "Not specified".
- If there are no action items, return an empty array.
- If there are no decisions, return an empty array.
- If there are no open questions, return an empty array.
- If there are no important topics, return an empty array.
- If there are no risks or blockers, return an empty array.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{transcript}")
        ]
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {
            "transcript": transcript
        }
    )

    # --------------------------------------------------------
    # Clean possible markdown fences
    # --------------------------------------------------------

    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]

    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        result = json.loads(response)

        return result

    except json.JSONDecodeError:

        print("\nWARNING: Mistral returned invalid JSON.")
        print("Raw response:")
        print(response)

        # Safe fallback
        return {
            "action_items": [],
            "key_decisions": [],
            "open_questions": [],
            "important_topics": [],
            "risks_and_blockers": []
        }


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_all(transcript: str) -> dict:

    return extract_meeting_information(transcript)