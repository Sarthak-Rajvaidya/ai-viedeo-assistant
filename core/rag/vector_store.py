import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from core.rag.embeddings import (
    embed_documents,
    get_embedding_dimension,
)

load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "meeting_transcripts"
)


# --------------------------------------------------
# Qdrant Client
# --------------------------------------------------

_client = None


def get_qdrant_client() -> QdrantClient:
    """
    Create and return a Qdrant Cloud client.
    The client is created only once and reused.
    """

    global _client

    if _client is None:

        if not QDRANT_URL:
            raise ValueError(
                "QDRANT_URL is not set. "
                "Check your .env file."
            )

        if not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_API_KEY is not set. "
                "Check your .env file."
            )

        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _client


# --------------------------------------------------
# Collection
# --------------------------------------------------

def ensure_collection() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    """

    client = get_qdrant_client()

    if client.collection_exists(COLLECTION_NAME):
        return

    dimension = get_embedding_dimension()

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=dimension,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created Qdrant collection: {COLLECTION_NAME}"
    )


# --------------------------------------------------
# Index transcript chunks
# --------------------------------------------------

def index_chunks(
    chunks: list[str],
    meeting_id: str,
    source_name: str = "meeting_transcript",
) -> int:
    """
    Convert transcript chunks into embeddings
    and store them in Qdrant Cloud.
    """

    if not chunks:
        raise ValueError(
            "No chunks provided for indexing."
        )

    if not meeting_id:
        raise ValueError(
            "meeting_id is required."
        )

    # Make sure collection exists
    ensure_collection()

    client = get_qdrant_client()

    # Generate embeddings using HuggingFace
    embeddings = embed_documents(chunks)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        # Deterministic ID
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{meeting_id}-{index}",
            )
        )

        payload = {
            "meeting_id": meeting_id,
            "chunk_id": index,
            "text": chunk,
            "source": source_name,
        }

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload,
        )

        points.append(point)

    # Upload vectors to Qdrant Cloud
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Indexed {len(points)} chunks into "
        f"Qdrant Cloud."
    )

    return len(points)