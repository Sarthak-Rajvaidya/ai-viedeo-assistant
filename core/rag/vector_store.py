import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from core.rag.embeddings import (
    embed_documents,
    get_embedding_dimension
)


QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "./qdrant_storage"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "meeting_transcripts"
)


_client = None


def get_qdrant_client():
    """
    Return persistent local Qdrant client.
    """

    global _client

    if _client is None:
        _client = QdrantClient(
            path=QDRANT_PATH
        )

    return _client


def ensure_collection():

    client = get_qdrant_client()

    if client.collection_exists(COLLECTION_NAME):
        return

    dimension = get_embedding_dimension()

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE
        )
    )

    print(
        f"Created Qdrant collection: "
        f"{COLLECTION_NAME}"
    )


def index_chunks(
    chunks: list[str],
    meeting_id: str,
    source_name: str = "meeting_transcript"
) -> int:

    if not chunks:
        raise ValueError("No chunks to index.")

    ensure_collection()

    client = get_qdrant_client()

    embeddings = embed_documents(chunks)

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{meeting_id}-{index}"
            )
        )

        payload = {
            "meeting_id": meeting_id,
            "chunk_id": index,
            "text": chunk,
            "source": source_name
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(
        f"Indexed {len(points)} chunks "
        f"into Qdrant."
    )

    return len(points)