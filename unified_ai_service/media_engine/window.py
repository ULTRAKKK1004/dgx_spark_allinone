"""무빙윈도우 유틸 — 미디어 청크링·last_frame·concat·crossfade."""
import asyncio
import os
import subprocess
import uuid
from pathlib import Path

from pydub import AudioSegment, silence


async def chunk_audio_fixed(
    audio_path: str,
    chunk_sec: float = 30,
    overlap_sec: float = 5,
    out_dir: str | None = None,
) -> list[Path]:
    """고정 길이 청크 + 오버랩 분할."""
    out_dir = out_dir or os.path.dirname(audio_path)
    os.makedirs(out_dir, exist_ok=True)
    audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
    total_ms = len(audio)
    chunk_ms = int(chunk_sec * 1000)
    step_ms = chunk_ms - int(overlap_sec * 1000)
    assert step_ms > 0, "overlap must be < chunk_sec"

    paths: list[Path] = []
    start = 0
    idx = 0
    while start < total_ms:
        end = min(start + chunk_ms, total_ms)
        chunk = audio[start:end]
        p = Path(out_dir) / f"chunk_{uuid.uuid4().hex[:6]}_{idx:03d}.wav"
        await asyncio.to_thread(chunk.export, p, format="wav")
        paths.append(p)
        if end >= total_ms:
            break
        start += step_ms
        idx += 1
    return paths


async def chunk_audio_smart(
    audio_path: str,
    target_range: tuple[float, float] = (4, 8),
    out_dir: str | None = None,
) -> list[Path]:
    """무음 기반 청크 (lecture_service.slice_audio 일반화)."""
    out_dir = out_dir or os.path.dirname(audio_path)
    os.makedirs(out_dir, exist_ok=True)
    audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
    min_ms, max_ms = int(target_range[0] * 1000), int(target_range[1] * 1000)

    silent_ranges = await asyncio.to_thread(
        silence.detect_silence, audio,
        min_silence_len=300,
        silence_thresh=audio.dBFS - 16,
    )
    split_points = [0]
    for s, e in silent_ranges:
        split_points.append((s + e) // 2)
    split_points.append(len(audio))

    final_points = [0]
    last = 0
    for p in split_points[1:]:
        d = p - last
        if d < min_ms:
            continue
        if d <= max_ms:
            final_points.append(p)
            last = p
        else:
            while (p - last) > max_ms:
                last += (min_ms + max_ms) // 2
                final_points.append(last)
            final_points.append(p)
            last = p
    if final_points[-1] < len(audio):
        final_points.append(len(audio))
    final_points = sorted(set(final_points))

    paths: list[Path] = []
    for i in range(len(final_points) - 1):
        s, e = final_points[i], final_points[i + 1]
        seg = audio[s:e]
        p = Path(out_dir) / f"smart_{uuid.uuid4().hex[:6]}_{i:03d}.wav"
        await asyncio.to_thread(seg.export, p, format="wav")
        paths.append(p)
    return paths


async def extract_last_frame(video_path: str, out_dir: str | None = None) -> Path:
    """비디오의 마지막 프레임을 JPEG로 저장."""
    out_dir = out_dir or os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir) / f"lastframe_{uuid.uuid4().hex[:6]}.jpg"
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
           "-update", "1", "-q:v", "2", str(out)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"extract_last_frame failed: {err.decode()}")
    return out


async def concat_videos(paths: list[str], output_path: str, overlap_frames: int = 0) -> Path:
    """ffmpeg concat demuxer로 비디오들을 이어붙임."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".txt")
    list_file.write_text("".join(f"file '{os.path.abspath(p)}'\n" for p in paths))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", str(out)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    list_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"concat_videos failed: {err.decode()}")
    return out


async def crossfade_audio_segments(
    paths: list[str], overlap_ms: int = 300, output_path: str | None = None
) -> Path:
    """pydub crossfade로 오디오 청크들을 부드럽게 잇는다."""
    if not paths:
        raise ValueError("paths is empty")
    segs = [await asyncio.to_thread(AudioSegment.from_file, p) for p in paths]
    out_seg = segs[0]
    for s in segs[1:]:
        out_seg = out_seg.append(s, crossfade=overlap_ms)
    out = Path(output_path) if output_path else Path(paths[0]).with_name(
        f"xf_{uuid.uuid4().hex[:6]}.wav"
    )
    await asyncio.to_thread(out_seg.export, out, format="wav")
    return out


async def get_media_duration(path: str) -> float:
    """ffprobe로 길이 초 단위 반환."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return float(stdout.decode().strip())
