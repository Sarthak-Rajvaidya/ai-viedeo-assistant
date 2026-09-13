from qdrant_client.models import Filter, FieldCondition, MatchValue

from core.rag.vector_store import (
    get_qdrant_client,
    ensure_collection,
    COLLECTION_NAME
)

from core.rag.embeddings import embed_query


def retrieve_chunks(
    query: str,
    meeting_id: str,
    top_k: int = 5
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    ensure_collection()

    client = get_qdrant_client()

    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="meeting_id",
                    match=MatchValue(
                        value=meeting_id
                    )
                )
            ]
        ),
        limit=top_k,
        with_payload=True
    ).points

    retrieved = []

    for result in results:

        payload = result.payload or {}

        retrieved.append(
            {
                "text": payload.get("text", ""),
                "chunk_id": payload.get("chunk_id"),
                "meeting_id": payload.get("meeting_id"),
                "score": result.score
            }
        )

    return retrieved