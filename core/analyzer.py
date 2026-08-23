from core.classifier import classify_content
from core.transcript_cleaner import clean_transcript
from core.summarize import summarize, generate_title
from core.extractor import extract_meeting_information


def analyze_transcript(transcript: str) -> dict:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    print("\n" + "=" * 70)
    print("STEP 1: TRANSCRIPT CLEANING")
    print("=" * 70)

    cleaned_transcript = clean_transcript(
        transcript
    )

    print("Transcript cleaning completed.")

    print("\n" + "=" * 70)
    print("STEP 2: CONTENT CLASSIFICATION")
    print("=" * 70)

    classification = classify_content(
        cleaned_transcript
    )

    print(
        f"Content Type: "
        f"{classification.get('content_type')}"
    )

    print(
        f"Confidence: "
        f"{classification.get('confidence')}"
    )

    print("\n" + "=" * 70)
    print("STEP 3: GENERATING TITLE")
    print("=" * 70)

    title = generate_title(
        cleaned_transcript
    )

    print(f"Title: {title}")

    print("\n" + "=" * 70)
    print("STEP 4: GENERATING SUMMARY")
    print("=" * 70)

    summary = summarize(
        cleaned_transcript
    )

    print("Summary generated.")

    print("\n" + "=" * 70)
    print("STEP 5: STRUCTURED INFORMATION EXTRACTION")
    print("=" * 70)

    extracted = extract_meeting_information(
        cleaned_transcript
    )

    print("Information extraction completed.")

    result = {
        "content_type": classification.get(
            "content_type",
            "other"
        ),

        "classification_confidence": classification.get(
            "confidence",
            0.0
        ),

        "title": title,

        "summary": summary,

        "action_items": extracted.get(
            "action_items",
            []
        ),

        "decisions": extracted.get(
            "decisions",
            []
        ),

        "open_questions": extracted.get(
            "open_questions",
            []
        ),

        "key_topics": extracted.get(
            "key_topics",
            []
        ),

        "raw_transcript": transcript,

        "cleaned_transcript": cleaned_transcript
    }

    return result