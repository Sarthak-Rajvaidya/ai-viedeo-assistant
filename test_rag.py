from core.rag.pipeline import (
    index_meeting,
    ask_meeting
)


TEST_TRANSCRIPT = """
The team discussed the deployment strategy for the AI Meeting Assistant.

Rahul suggested using AWS ECS for production deployment.

The team agreed that Docker would be used to package the application.

Sarthak will prepare the Docker configuration by Friday.

The team decided to use Qdrant as the vector database for the RAG system.

The first version will use HuggingFace embeddings locally.

The team will evaluate Gemini embeddings later if retrieval quality needs improvement.

The backend will continue using Mistral AI for response generation.
"""


print("=" * 60)
print("INDEXING TEST MEETING")
print("=" * 60)

result = index_meeting(
    TEST_TRANSCRIPT
)

meeting_id = result["meeting_id"]

print(
    f"Meeting ID: {meeting_id}"
)

print(
    f"Chunks indexed: "
    f"{result['chunk_count']}"
)


print("\n" + "=" * 60)
print("TESTING RETRIEVAL + GENERATION")
print("=" * 60)


questions = [
    "What deployment strategy did the team choose?",
    "Who will prepare the Docker configuration?",
    "Which vector database was selected?",
    "Why are they considering Gemini embeddings?"
]


for question in questions:

    print("\nQUESTION:")
    print(question)

    response = ask_meeting(
        question=question,
        meeting_id=meeting_id
    )

    print("\nANSWER:")
    print(response["answer"])

    print("\nSOURCES:")

    for source in response["sources"]:

        print(
            f"- Chunk {source['chunk_id']} "
            f"(score={source['score']:.4f})"
        )