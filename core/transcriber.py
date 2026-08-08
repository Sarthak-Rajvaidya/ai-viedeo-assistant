import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL","small")

_modle = None

def load_model ():
    global_model
    
    if _modle is None:
        print(f"Loading model....")
        
        _modle = whisper.load_model(WHISPER_MODEL)
        
        print("whisper model loaded successfully")
        
    return _modle

def transcribe_chunk(chunk_path : str,translate : bool = False):
    model = load_model()
    
    task = "translate" if translate else "transcribe"
    
    result = model.transcribe(chunk_path,task = task)
    
    return result['text']

def transcribe_all(chunks : list,translate : bool = False):
    
    full_transcript = ""
    
    for i , chunk in enumerate(chunks):
        print("Transcribing chunk {i+1}")
        
        text = transcribe_chunk(chunk,translate=translate)
        
        
        full_transcript += text + " "
        
    print("Transcription completed")
    
    return full_transcript
    