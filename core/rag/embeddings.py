from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """

    global _model

    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}")

        _model = SentenceTransformer(MODEL_NAME)

        print("Embedding model loaded successfully.")

    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for document chunks.
    """

    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user question.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding.tolist()


def get_embedding_dimension() -> int:
    """
    Return vector dimension required by Qdrant.
    """

    model = get_embedding_model()

    return model.get_embedding_dimension()