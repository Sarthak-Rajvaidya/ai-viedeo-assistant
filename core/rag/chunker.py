from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_transcript(
    transcript: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> list[str]:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(transcript)

    return chunks