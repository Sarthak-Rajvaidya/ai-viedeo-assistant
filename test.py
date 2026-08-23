from utils.audio_processor import process_input

from core.transcriber import transcribe_all
from core.sarvam_transcriber import transcribe_sarvam_batch

from core.summarize import summarize, generate_title

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = "https://youtu.be/sHpgX2FnToM?si=DUBOWBOC4gbdKBv_"

# Choose transcription engine:
#
# "whisper" -> OpenAI Whisper (local)
# "sarvam"  -> Sarvam AI Saaras v3
#
# For your Hindi video, use "sarvam".
TRANSCRIPTION_ENGINE = "sarvam"


# ============================================================
# STEP 1: AUDIO PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: AUDIO PROCESSING")
print("=" * 70)

audio_data = process_input(SOURCE)

wav_path = audio_data["wav_path"]
chunks = audio_data["chunks"]

print(f"\nWAV file: {wav_path}")
print(f"Number of chunks: {len(chunks)}")


# ============================================================
# STEP 2: SPEECH-TO-TEXT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: SPEECH-TO-TEXT")
print("=" * 70)


if TRANSCRIPTION_ENGINE == "whisper":

    print("\nUsing OpenAI Whisper...\n")

    transcript = transcribe_all(chunks)


elif TRANSCRIPTION_ENGINE == "sarvam":

    print("\nUsing Sarvam AI Saaras v3...\n")

    # Sarvam Batch API can process the complete WAV file.
    transcript = transcribe_sarvam_batch(wav_path)


else:

    raise ValueError(
        "Invalid transcription engine. "
        "Use either 'whisper' or 'sarvam'."
    )


print("\n========== FULL TRANSCRIPT ==========\n")
print(transcript)


# ============================================================
# STEP 3: GENERATE MEETING TITLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: GENERATING MEETING TITLE")
print("=" * 70)

title = generate_title(transcript)

print("\n========== MEETING TITLE ==========\n")
print(title)


# ============================================================
# STEP 4: GENERATE MEETING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: GENERATING MEETING SUMMARY")
print("=" * 70)

summary = summarize(transcript)

print("\n========== MEETING SUMMARY ==========\n")
print(summary)


# ============================================================
# STEP 5: EXTRACT ACTION ITEMS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: EXTRACTING ACTION ITEMS")
print("=" * 70)

action_items = extract_action_items(transcript)

print("\n========== ACTION ITEMS ==========\n")
print(action_items)


# ============================================================
# STEP 6: EXTRACT KEY DECISIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: EXTRACTING KEY DECISIONS")
print("=" * 70)

decisions = extract_key_decisions(transcript)

print("\n========== KEY DECISIONS ==========\n")
print(decisions)


# ============================================================
# STEP 7: EXTRACT OPEN QUESTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: EXTRACTING OPEN QUESTIONS")
print("=" * 70)

questions = extract_questions(transcript)

print("\n========== OPEN QUESTIONS ==========\n")
print(questions)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MEETING INTELLIGENCE PIPELINE COMPLETED")
print("=" * 70)

print("\n\nTITLE")
print("-" * 70)
print(title)

print("\n\nSUMMARY")
print("-" * 70)
print(summary)

print("\n\nACTION ITEMS")
print("-" * 70)
print(action_items)

print("\n\nKEY DECISIONS")
print("-" * 70)
print(decisions)

print("\n\nOPEN QUESTIONS")
print("-" * 70)
print(questions)

print("\n" + "=" * 70)
print("END")
print("=" * 70)