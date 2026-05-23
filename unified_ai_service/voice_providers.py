"""TTS provider abstraction with optional ElevenLabs fallback."""
from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydub import AudioSegment


BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEFAULT_ELEVENLABS_VOICE_ID = "airYK6ydeWdrJg6gyZA3"

load_dotenv("/home/yanus/.env")


@dataclass(frozen=True)
class TTSChunk:
    index: int
    text: str
    context_before: str = ""
    context_after: str = ""


def _public_url(path: str) -> str:
    return f"/api/results/{os.path.basename(path)}"


def local_f5_available() -> bool:
    return shutil.which("f5-tts_infer-cli") is not None


def get_elevenlabs_voice_id(voice: str = "default") -> str:
    if voice and voice != "default":
        return voice
    return os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID


def choose_provider(requested: str = "auto", quality: str = "standard") -> str:
    requested = requested or "auto"
    if requested not in {"auto", "local_f5", "elevenlabs"}:
        raise ValueError("provider must be auto, local_f5, or elevenlabs")
    if requested != "auto":
        return requested
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY") and get_elevenlabs_voice_id())
    if has_elevenlabs and (quality == "high" or not local_f5_available()):
        return "elevenlabs"
    return "local_f5"


def split_text_for_tts(text: str, limit: int = 2500) -> list[TTSChunk]:
    normalized = text.strip()
    if not normalized:
        return [TTSChunk(index=0, text="")]
    parts = re.findall(r"\S+\s*", normalized)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        candidate = f"{current}{part}" if current else part
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) <= limit:
            current = part
        else:
            for start in range(0, len(part), limit):
                chunks.append(part[start : start + limit])
            current = ""
    if current:
        chunks.append(current)

    out: list[TTSChunk] = []
    for i, chunk in enumerate(chunks):
        before = chunks[i - 1][-300:] if i > 0 else ""
        after = chunks[i + 1][:300] if i + 1 < len(chunks) else ""
        out.append(TTSChunk(index=i, text=chunk, context_before=before, context_after=after))
    return out


async def synthesize_speech(
    text: str,
    provider: str = "auto",
    quality: str = "standard",
    voice: str = "default",
    ref_audio: str = "",
    ref_text: str = "",
    output_path: str | None = None,
) -> dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    selected = choose_provider(provider, quality)
    if not output_path:
        ext = "mp3" if selected == "elevenlabs" else "wav"
        output_path = os.path.join(RESULTS_DIR, f"tts_{uuid.uuid4().hex[:8]}.{ext}")

    try:
        if selected == "elevenlabs":
            path = await _synthesize_elevenlabs(text, output_path, voice=voice)
        else:
            path = await _synthesize_local_f5(text, ref_audio, ref_text, output_path)
        return {"provider": selected, "path": path, "url": _public_url(path), "characters": len(text)}
    except Exception:
        if provider == "auto" and selected == "elevenlabs":
            fallback_path = str(Path(output_path).with_suffix(".wav"))
            path = await _synthesize_local_f5(text, ref_audio, ref_text, fallback_path)
            return {"provider": "local_f5", "path": path, "url": _public_url(path), "characters": len(text)}
        raise


async def _synthesize_local_f5(text: str, ref_audio: str, ref_text: str, output_path: str) -> str:
    import media_audio

    chunks = split_text_for_tts(text, limit=900)
    if len(chunks) == 1:
        return await media_audio.generate_tts_with_effects(chunks[0].text, ref_audio, ref_text, output_path)

    tmp_dir = os.path.join(BASE_DIR, "tmp_tts")
    os.makedirs(tmp_dir, exist_ok=True)
    segments: list[AudioSegment] = []
    temp_paths: list[str] = []
    try:
        for chunk in chunks:
            chunk_path = os.path.join(tmp_dir, f"tts_chunk_{uuid.uuid4().hex[:8]}_{chunk.index}.wav")
            temp_paths.append(chunk_path)
            await media_audio.generate_tts_with_effects(chunk.text, ref_audio, ref_text, chunk_path)
            segments.append(AudioSegment.from_wav(chunk_path))
        combined = AudioSegment.empty()
        for i, segment in enumerate(segments):
            if i:
                combined += AudioSegment.silent(duration=180)
            combined += segment
        combined.export(output_path, format="wav")
        return output_path
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


async def _synthesize_elevenlabs(text: str, output_path: str, voice: str = "default") -> str:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = get_elevenlabs_voice_id(voice)
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not configured")

    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    chunks = split_text_for_tts(text, limit=2500)
    tmp_paths: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            for chunk in chunks:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
                params = {"output_format": output_format}
                payload = {
                    "text": chunk.text,
                    "model_id": model_id,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                }
                response = await client.post(
                    url,
                    params=params,
                    headers={"xi-api-key": api_key, "accept": "audio/mpeg"},
                    json=payload,
                )
                response.raise_for_status()
                part_path = os.path.join(RESULTS_DIR, f"eleven_{uuid.uuid4().hex[:8]}_{chunk.index}.mp3")
                Path(part_path).write_bytes(response.content)
                tmp_paths.append(part_path)

        if len(tmp_paths) == 1:
            os.replace(tmp_paths[0], output_path)
            return output_path

        combined = AudioSegment.empty()
        for i, part_path in enumerate(tmp_paths):
            if i:
                combined += AudioSegment.silent(duration=180)
            combined += AudioSegment.from_file(part_path)
        combined.export(output_path, format=Path(output_path).suffix.lstrip(".") or "mp3")
        return output_path
    finally:
        for path in tmp_paths:
            if path != output_path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


async def list_elevenlabs_voices() -> list[dict[str, Any]]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return []

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
        )
        response.raise_for_status()
    voices = response.json().get("voices", [])
    compact: list[dict[str, Any]] = []
    for voice in voices:
        compact.append(
            {
                "voice_id": voice.get("voice_id", ""),
                "name": voice.get("name", "Unnamed voice"),
                "category": voice.get("category", ""),
                "labels": voice.get("labels") or {},
            }
        )
    return compact
