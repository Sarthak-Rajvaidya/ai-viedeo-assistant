import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

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
# Collection + Payload Index
# --------------------------------------------------

def ensure_collection() -> None:
    """
    Make sure the Qdrant collection exists.

    Also creates a keyword payload index on
    meeting_id because our retrieval queries
    filter by meeting_id.
    """

    client = get_qdrant_client()

    # ----------------------------------------------
    # Create collection if it doesn't exist
    # ----------------------------------------------

    if not client.collection_exists(COLLECTION_NAME):

        dimension = get_embedding_dimension()

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{COLLECTION_NAME}"
        )

    # ----------------------------------------------
    # Create meeting_id payload index
    # ----------------------------------------------

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="meeting_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print(
        "Ensured payload index: meeting_id"
    )


# --------------------------------------------------
# Index Transcript Chunks
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

    # Make sure collection and index exist
    ensure_collection()

    client = get_qdrant_client()

    # Generate HuggingFace embeddings
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

        # Deterministic point ID
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

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # Upload points to Qdrant Cloud
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print(
        f"Indexed {len(points)} chunks "
        f"into Qdrant Cloud."
    )

    return len(points)