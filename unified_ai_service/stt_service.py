import whisper
import os

# Load model lazily
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model("base") # Use "base" for faster inference
    return _model

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = get_model()
    result = model.transcribe(audio_path)
    return result.get("text", "")
