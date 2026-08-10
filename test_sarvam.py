from utils.audio_processor import process_input
from core.sarvam_transcriber import transcribe_sarvam_batch


source = "https://youtu.be/OnVFOjvjK5I"

# Process input
audio_data = process_input(source)

# Get the FULL WAV file
wav_path = audio_data["wav_path"]

print("\nStarting Sarvam Batch transcription...\n")

transcript = transcribe_sarvam_batch(wav_path)

print("\n========== FULL TRANSCRIPT ==========\n")

print(transcript)