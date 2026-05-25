import os
from faster_whisper import WhisperModel

# Constants
MODEL_SIZE = "base"
# Default to CPU for stability as some environments lack CUDA-enabled CTranslate2
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Load model lazily
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading Faster-Whisper model ({MODEL_SIZE}) on {DEVICE}...")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model

async def transcribe_audio(audio_path: str, language: str = None) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model = get_model()
    # transcribe is synchronous in faster-whisper, but we wrap it in a thread if needed
    # for simplicity in this service, we run it directly or use asyncio.to_thread
    import asyncio
    
    def _run():
        segments, info = model.transcribe(audio_path, beam_size=5, language=language)
        text = "".join(segment.text for segment in segments)
        return text.strip()

    return await asyncio.to_thread(_run)

async def transcribe_with_timestamps(audio_path: str, language: str = None):
    """Returns segments with timestamps for subtitle generation."""
    model = get_model()
    import asyncio
    
    def _run():
        segments, info = model.transcribe(audio_path, beam_size=5, language=language)
        results = []
        for s in segments:
            results.append({
                "start": s.start,
                "end": s.end,
                "text": s.text.strip()
            })
        return results

    return await asyncio.to_thread(_run)

def format_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_from_segments(segments: list[dict]) -> str:
    """Converts segments with start/end/text to SRT content."""
    lines = []
    for i, s in enumerate(segments):
        lines.append(str(i + 1))
        start = format_timestamp(s["start"])
        end = format_timestamp(s["end"])
        lines.append(f"{start} --> {end}")
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)
