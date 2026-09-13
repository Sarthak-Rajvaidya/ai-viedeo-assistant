from core.rag.pipeline import index_meeting
from core.rag.retriever import retrieve_chunks


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
print("CREATING MEETING INDEX")
print("=" * 60)

result = index_meeting(TEST_TRANSCRIPT)

meeting_id = result["meeting_id"]

print(f"Meeting ID: {meeting_id}")
print(f"Chunks indexed: {result['chunk_count']}")


questions = [
    "What deployment strategy did the team choose?",
    "Who will prepare the Docker configuration?",
    "Which vector database was selected?",
    "Why are they considering Gemini embeddings?"
]


for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(question)

    results = retrieve_chunks(
        query=question,
        meeting_id=meeting_id,
        top_k=3
    )

    print("\nRETRIEVED CHUNKS")

    for result in results:

        print(
            f"\nChunk ID: {result['chunk_id']}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Text:\n{result['text']}"
        )