from utils.audio_processor import process_input
from core.transcriber import transcribe_all


source = "https://www.youtube.com/watch?v=F8NKVhkZZWI"

audio_data = process_input(source)

chunks = audio_data["chunks"]

print(
    transcribe_all(chunks)
)