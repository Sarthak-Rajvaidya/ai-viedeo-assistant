import os
import json

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.sarvam_transcriber import transcribe_sarvam_batch
from core.analyzer import analyze_transcript


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = "https://youtu.be/sHpgX2FnToM?si=DUBOWBOC4gbdKBv_"


# ============================================================
# STEP 1: AUDIO PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: AUDIO PROCESSING")
print("=" * 70)

audio_data = process_input(
    SOURCE
)

wav_path = audio_data["wav_path"]

print(f"\nWAV file: {wav_path}")

print(
    f"Number of chunks: "
    f"{len(audio_data.get('chunks', []))}"
)


# ============================================================
# STEP 2: SPEECH-TO-TEXT / TRANSLATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: SPEECH-TO-TEXT")
print("=" * 70)

print("\nUsing Sarvam AI Saaras v3...")

raw_transcript = transcribe_sarvam_batch(
    wav_path
)

print("\n========== RAW TRANSCRIPT ==========\n")

print(raw_transcript)


# ============================================================
# STEP 3: AI ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: AI MEETING INTELLIGENCE")
print("=" * 70)

result = analyze_transcript(
    raw_transcript
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


print("\nTITLE")
print("-" * 70)

print(
    result["title"]
)


print("\nCONTENT TYPE")
print("-" * 70)

print(
    result["content_type"]
)

print(
    f"Confidence: "
    f"{result['classification_confidence']}"
)


print("\nSUMMARY")
print("-" * 70)

print(
    result["summary"]
)


print("\nACTION ITEMS")
print("-" * 70)

if result["action_items"]:

    for index, item in enumerate(
        result["action_items"],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item.get('task', 'Not specified')}"
        )

        print(
            f"   Owner: "
            f"{item.get('owner', 'Not specified')}"
        )

        print(
            f"   Deadline: "
            f"{item.get('deadline', 'Not specified')}"
        )

        print(
            f"   Priority: "
            f"{item.get('priority', 'Not specified')}"
        )

else:

    print("No action items found.")


print("\nKEY DECISIONS")
print("-" * 70)

if result["decisions"]:

    for index, item in enumerate(
        result["decisions"],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item.get('decision', 'Not specified')}"
        )

        print(
            f"   Made by: "
            f"{item.get('made_by', 'Not specified')}"
        )

        print(
            f"   Reason: "
            f"{item.get('reason', 'Not specified')}"
        )

else:

    print("No key decisions found.")


print("\nOPEN QUESTIONS")
print("-" * 70)

if result["open_questions"]:

    for index, item in enumerate(
        result["open_questions"],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item.get('question', 'Not specified')}"
        )

        print(
            f"   Context: "
            f"{item.get('context', 'Not specified')}"
        )

else:

    print("No open questions found.")


print("\nKEY TOPICS")
print("-" * 70)

if result["key_topics"]:

    for index, item in enumerate(
        result["key_topics"],
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item.get('topic', 'Not specified')}"
        )

        print(
            f"   Description: "
            f"{item.get('description', '')}"
        )

else:

    print("No key topics found.")


# ============================================================
# SAVE STRUCTURED RESULT
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)

output_file = (
    "reports/meeting_analysis.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=4
    )


print("\n" + "=" * 70)
print("ANALYSIS SAVED")
print("=" * 70)

print(
    f"Saved to: {output_file}"
)


print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)