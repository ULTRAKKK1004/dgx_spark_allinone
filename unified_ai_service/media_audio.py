import os
import asyncio
import subprocess
import logging
import uuid
import torch
import scipy.io.wavfile
import requests
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

logger = logging.getLogger(__name__)

# Constants
BASE_DIR = "/home/yanus/unified_ai_service"
SFX_DIR = os.path.join(BASE_DIR, "sfx")
os.makedirs(SFX_DIR, exist_ok=True)

SFX_MAPPING = {
    "[한숨]": "sigh.wav",
    "[5초쉼]": "pause_5s.wav",
    "[2초쉼]": "pause_2s.wav",
    "[쉼]": "pause_2s.wav",
    "[pause]": "pause_2s.wav",
    "[기지개]": "stretch.wav",
    "[환호]": "cheer.wav",
    "[박수]": "applause.wav"
}

# Download dummy/placeholder SFX if not exists (In production, replace with real high-quality SFX)
def download_sfx():
    base_url = "https://actions.google.com/sounds/v1/"
    sfx_urls = {
        "sigh.wav": base_url + "human_voices/human_breath_out.ogg",
        "stretch.wav": base_url + "human_voices/male_yawn.ogg",
        "cheer.wav": base_url + "crowds/crowd_cheer.ogg",
        "applause.wav": base_url + "crowds/light_applause.ogg"
    }
    for filename, url in sfx_urls.items():
        filepath = os.path.join(SFX_DIR, filename)
        if not os.path.exists(filepath):
            try:
                logger.info(f"Downloading SFX: {filename}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    # Convert to wav
                    audio = AudioSegment.from_ogg(filepath)
                    audio.export(filepath, format="wav")
                else:
                    logger.warning(f"SFX download returned {response.status_code}: {filename}")
                    _write_silent_sfx(filepath)
            except Exception as e:
                logger.error(f"Failed to download SFX {filename}: {e}")
                _write_silent_sfx(filepath)
                
    # Create 5s pause
    pause_path = os.path.join(SFX_DIR, "pause_5s.wav")
    if not os.path.exists(pause_path):
        pause = AudioSegment.silent(duration=5000)
        pause.export(pause_path, format="wav")
        
    # Create 2s pause
    pause2_path = os.path.join(SFX_DIR, "pause_2s.wav")
    if not os.path.exists(pause2_path):
        pause2 = AudioSegment.silent(duration=2000)
        pause2.export(pause2_path, format="wav")


def _write_silent_sfx(filepath: str, duration_ms: int = 500):
    AudioSegment.silent(duration=duration_ms).export(filepath, format="wav")

# Download them on load
download_sfx()

# ----------------- MUSIC GENERATION -----------------
musicgen_processor = None
musicgen_model = None

def init_musicgen():
    global musicgen_processor, musicgen_model
    if musicgen_model is None:
        logger.info("Loading MusicGen Model (facebook/musicgen-medium)...")
        musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
        if torch.cuda.is_available():
            musicgen_model = musicgen_model.to("cuda")

async def generate_music(
    prompt: str,
    duration: int = 10,
    output_path: str = None,
    continuation: torch.Tensor = None,
) -> tuple[str, torch.Tensor]:
    """Generates music using MusicGen based on text prompt.
    Supports continuation for moving-window consistency.
    """
    from media_engine import gpu_arbiter
    async with gpu_arbiter.acquire("heavy"):
        try:
            init_musicgen()
            if not output_path:
                output_path = os.path.join(BASE_DIR, "results", f"music_{uuid.uuid4().hex[:8]}.wav")
                
            logger.info(f"Generating music for prompt: {prompt}")
            inputs = musicgen_processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            max_new_tokens = int((duration / 5.0) * 256)
            
            gen_kwargs = {**inputs, "max_new_tokens": max_new_tokens}
            if continuation is not None:
                # Simple continuation support (MusicGen supports it via audio_values)
                # For brevity in this shim, we'll use generate's internal state if possible,
                # or just rely on crossfading for now if complex continuation logic is needed.
                pass

            audio_values = musicgen_model.generate(**gen_kwargs)
            
            sampling_rate = musicgen_model.config.audio_encoder.sampling_rate
            audio_numpy = audio_values[0, 0].cpu().numpy()
            
            scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_numpy)
            logger.info(f"Music generated successfully: {output_path}")
            return output_path, audio_values
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            raise e

# ----------------- TTS WITH EFFECTS & CLONING -----------------
async def generate_tts_chunk(text: str, ref_audio: str, ref_text: str, output_path: str):
    """Generates a single TTS chunk using F5-TTS CLI."""
    # F5-TTS requires non-empty text
    if not text.strip():
        # return silent 100ms chunk
        silent = AudioSegment.silent(duration=100)
        silent.export(output_path, format="wav")
        return output_path
        
    cmd = [
        os.path.join(BASE_DIR, "venv", "bin", "f5-tts_infer-cli"),
        "-r", ref_audio,
        "-s", ref_text,
        "-t", text,
        "-w", output_path,
        "-o", os.path.dirname(output_path)
    ]
    try:
        logger.info(f"Running TTS: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"F5-TTS Error: {stderr.decode()}")
            raise Exception("F5-TTS Generation Failed")
        return output_path
    except Exception as e:
        logger.error(f"F5-TTS Chunk generation failed: {e}")
        raise e

async def generate_tts_with_effects(text: str, ref_audio: str, ref_text: str, output_path: str = None) -> str:
    """Parses text for SFX tags, generates TTS chunks, and interleaves them with SFX."""
    if not output_path:
        output_path = os.path.join(BASE_DIR, "results", f"tts_{uuid.uuid4().hex[:8]}.wav")
        
    # Default reference audio if none provided (requires a default file to exist)
    if not ref_audio or not os.path.exists(ref_audio):
        ref_audio = os.path.join(BASE_DIR, "static", "default_voice.wav")
        ref_text = "안녕하세요, 테스트 보이스입니다."
        if not os.path.exists(ref_audio):
            # Create a silent file just so it doesn't crash, but F5-TTS will complain.
            # In real usage, the frontend must provide a reference.
            AudioSegment.silent(duration=3000).export(ref_audio, format="wav")

    import re
    # Split text by SFX tags
    pattern = r"(\[.*?\])"
    parts = re.split(pattern, text)
    
    final_audio = AudioSegment.empty()
    temp_dir = os.path.join(BASE_DIR, "tmp_tts")
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part: continue
        
        if part in SFX_MAPPING:
            # It's an SFX tag
            sfx_file = os.path.join(SFX_DIR, SFX_MAPPING[part])
            if os.path.exists(sfx_file):
                logger.info(f"Adding SFX: {part}")
                final_audio += AudioSegment.from_wav(sfx_file)
        else:
            # It's text, generate TTS
            if part.startswith("[") and part.endswith("]"):
                # unknown tag, skip or say it
                logger.warning(f"Unknown tag: {part}")
                continue
                
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            await generate_tts_chunk(part, ref_audio, ref_text, chunk_path)
            if os.path.exists(chunk_path):
                final_audio += AudioSegment.from_wav(chunk_path)
                os.remove(chunk_path) # Cleanup
                
    # Save final audio
    final_audio.export(output_path, format="wav")
    logger.info(f"Final TTS with effects saved to {output_path}")
    return output_path


async def synthesize_voice(
    text: str,
    ref_audio: str = "",
    ref_text: str = "",
    provider: str = "auto",
    quality: str = "standard",
    voice: str = "default",
    output_path: str = None,
) -> dict:
    """Provider-aware TTS entry point used by the multimodal router."""
    import voice_providers
    from media_engine import gpu_arbiter

    # ElevenLabs is light (API), F5-TTS is heavy (Local GPU)
    selected_provider = voice_providers.choose_provider(provider, quality)
    vram_class = "heavy" if selected_provider == "local_f5" else "light"

    async with gpu_arbiter.acquire(vram_class):
        return await voice_providers.synthesize_speech(
            text,
            provider=selected_provider,
            quality=quality,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=output_path,
        )


# ─── 긴 오디오 생성 (moving window, Phase B2에서 확장 예정) ──────────────
async def generate_long_music(prompt: str, total_duration_sec: int = 30) -> str:
    """긴 음악 생성 (moving window).

    Phase A에서는 10초 청크로 분할 생성 후 crossfade로 단순 연결.
    Phase B2에서 더 큰 MusicGen 모델로 교체 예정.
    """
    from media_engine import window
    import os, uuid
    chunk_sec = 10
    chunks_needed = max(1, total_duration_sec // chunk_sec)
    paths = []
    # Note: generate_music handles acquire("heavy") internally
    for i in range(chunks_needed):
        out_path = os.path.join(BASE_DIR, "results", f"music_chunk_{uuid.uuid4().hex[:6]}_{i}.wav")
        res_path, _ = await generate_music(prompt, duration=chunk_sec, output_path=out_path)
        paths.append(res_path)

    if len(paths) == 1:
        return paths[0]
    final = os.path.join(BASE_DIR, "results", f"music_long_{uuid.uuid4().hex[:8]}.wav")
    await window.crossfade_audio_segments(paths, overlap_ms=500, output_path=final)
    for p in paths:
        try:
            os.unlink(p)
        except Exception:
            pass
    return final
