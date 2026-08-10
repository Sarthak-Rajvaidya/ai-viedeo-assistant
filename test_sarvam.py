from utils.audio_processor import process_input
from core.sarvam_transcriber import transcribe_sarvam_batch

source = "https://youtu.be/OnVFOjvjK5I"

# Process YouTube URL
chunks = process_input(source)

# For testing, use the first generated WAV chunk
audio_path = chunks[0]

print("\nStarting Sarvam Batch transcription...\n")

transcript = transcribe_sarvam_batch(audio_path)

print("\n========== TRANSCRIPT ==========\n")
print(transcript)