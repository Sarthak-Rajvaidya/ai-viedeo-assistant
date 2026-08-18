#Actionable items - eg google meet - u have to do that u hvae to do it 

#decisions, questions 

from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough,RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest",mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature = 0.2) #temp for creative response

def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) | ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{text}"),
    ]) | llm | StrOutputParser()
        )


def extract_action_items(transcript: str) -> str:

    chain = build_chain(

        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all action items.\n\n"

        "For each action item provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, otherwise write 'Not specified')\n\n"

        "Important rules:\n"
        "- Only extract action items explicitly mentioned in the transcript.\n"
        "- Do not invent tasks, owners, or deadlines.\n"
        "- If the owner is not mentioned, write 'Not specified'.\n"
        "- If no action items exist, say 'No action items found.'\n\n"

        "Format the result as a numbered list."
    )

    return chain.invoke(transcript)




def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all important decisions "
        "that were explicitly made during the meeting.\n\n"

        "For each decision:\n"
        "- Clearly describe what was decided.\n"
        "- Include the reason or context if it is explicitly mentioned.\n"
        "- Include the decision owner or responsible person if explicitly mentioned.\n\n"

        "Important rules:\n"
        "- Only extract decisions explicitly supported by the transcript.\n"
        "- Do not infer or invent decisions.\n"
        "- Do not treat suggestions, opinions, or possibilities as decisions.\n"
        "- If no clear decisions were made, say 'No key decisions found.'\n\n"

        "Format the result as a numbered list."
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all unresolved questions, "
        "pending issues, or topics that require follow-up.\n\n"

        "For each question or issue:\n"
        "- Clearly describe the question or unresolved issue.\n"
        "- Include the person responsible for answering it if explicitly mentioned.\n"
        "- Include a deadline if explicitly mentioned.\n\n"

        "Important rules:\n"
        "- Only extract questions or unresolved issues supported by the transcript.\n"
        "- Do not invent questions.\n"
        "- Do not include questions that were already answered during the meeting.\n"
        "- Do not treat general discussion as an unresolved question.\n"
        "- If no open questions exist, say 'No open questions found.'\n\n"

        "Format the result as a numbered list."
    )

    return chain.invoke(transcript)