from utils.audio_processor import process_input
from core.sarvam_transcriber import transcribe_sarvam_batch


source = "https://youtu.be/OnVFOjvjK5I"

# Process YouTube input
audio_data = process_input(source)

# Get full WAV file
wav_path = audio_data["wav_path"]

print("\nStarting Sarvam Hindi → English translation...\n")

transcript = transcribe_sarvam_batch(wav_path)

print("\n========== ENGLISH TRANSCRIPT ==========\n")

print(transcript)