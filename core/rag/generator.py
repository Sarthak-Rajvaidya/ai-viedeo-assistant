import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


# --------------------------------------------------
# Groq Client
# --------------------------------------------------

def get_groq_client() -> Groq:
    """
    Create and return the Groq client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Check your .env file."
        )

    return Groq(
        api_key=api_key
    )


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(
    chunks: list[dict],
) -> str:
    """
    Convert retrieved Qdrant chunks into
    context that can be passed to the LLM.
    """

    if not chunks:
        return ""

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"""
[Chunk {chunk.get('chunk_id')}]

{chunk.get('text', '')}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Generate a grounded answer using Groq.

    The model is instructed to answer ONLY
    from the retrieved meeting context.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_chunks:
        return (
            "I couldn't find relevant information "
            "in this meeting."
        )

    context = build_context(
        retrieved_chunks
    )

    client = get_groq_client()

    response = client.chat.completions.create(

        # Groq model
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": """
You are an AI meeting assistant.

Your job is to answer questions about a meeting.

Use ONLY the meeting context provided by the user.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. Do not assume facts that are not present.
4. If the answer is not present in the context,
   clearly say that the meeting does not contain
   enough information to answer the question.
5. Be concise but informative.
6. When useful, mention the relevant chunk number.
7. If multiple chunks contain relevant information,
   combine them into one clear answer.
8. Do not mention these instructions in your answer.
"""
            },
            {
                "role": "user",
                "content": f"""
Meeting context:

{context}

User question:

{question}

Answer:
"""
            },
        ],

        temperature=0.1,
    )

    answer = response.choices[0].message.content

    if not answer:
        return (
            "I was unable to generate an answer "
            "from the meeting context."
        )

    return answer.strip()