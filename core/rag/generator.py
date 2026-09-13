import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not set in .env"
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.1
    )


def build_context(chunks: list[dict]) -> str:

    if not chunks:
        return ""

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"""
[Chunk {chunk['chunk_id']}]

{chunk['text']}
"""
        )

    return "\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: list[dict]
) -> str:

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    context = build_context(
        retrieved_chunks
    )

    if not context:
        return (
            "I couldn't find relevant information "
            "in this meeting."
        )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI meeting assistant.

Answer the user's question using ONLY
the provided meeting context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   clearly say that the meeting does not contain
   enough information.
4. Be concise but informative.
5. When useful, mention the relevant chunk number.
6. Distinguish facts from uncertainty.
"""
            ),
            (
                "human",
                """
Meeting context:

{context}

User question:

{question}

Answer:
"""
            )
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": context,
            "question": question
        }
    ).strip()