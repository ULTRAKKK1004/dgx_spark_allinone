"""비디오 생성·편집·분석 (media_engine 사용하는 얇은 shim).

generate_long_video: Wan2.2 i2v로 moving-window 방식 긴 비디오 생성.
edit_video: moviepy 기반 (오디오 덧입히기 / 이미지 append).
shorten_video: 기본 자르기 + 세로 크롭.
analyze_video: 다중 키프레임을 VLM에 전달.
"""
import base64
import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from multimodal_executor import ExecutionContext

from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips

import llm_service
from media_engine import runner, comfyui_client, window

logger = logging.getLogger(__name__)

BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TMP_DIR = os.path.join(BASE_DIR, "tmp_video")
COMFY_INPUT_DIR = os.getenv("COMFYUI_INPUT_DIR", "/home/yanus/Docker/input")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


# ─── 비디오 생성 (i2v moving window) ─────────────────────────────────────
async def generate_long_video(
    prompt: str,
    base_image_path: str,
    total_duration_target: int = 30,
    fps: int = 16,
) -> Path:
    """Wan2.2 i2v로 5초 청크를 moving window로 이어붙여 긴 비디오 생성."""
    logger.info("Long video gen: %ds target for %s", total_duration_target, prompt)

    chunk_sec = 5
    chunks_needed = max(1, total_duration_target // chunk_sec)
    chunk_paths: list[Path] = []
    current_image = base_image_path

    for i in range(chunks_needed):
        filename = f"i2v_{uuid.uuid4().hex[:6]}_{i}.png"
        await comfyui_client.upload_image(current_image, filename)

        chunk = await runner.run(
            "video.i2v.wan22",
            prompt=prompt,
            image_name=filename,
            frames=int(chunk_sec * fps + 1),
        )
        chunk_paths.append(chunk)

        if i < chunks_needed - 1:
            next_frame = await window.extract_last_frame(str(chunk), out_dir=TMP_DIR)
            current_image = str(next_frame)

    out = Path(RESULTS_DIR) / f"long_vid_{uuid.uuid4().hex[:8]}.mp4"
    await window.concat_videos([str(p) for p in chunk_paths], str(out))
    return out


# ─── 음성 기반 강의/발화 비디오 생성 (S2V moving window) ───────────────
async def generate_talking_video(
    prompt: str,
    base_image_path: str,
    audio_path: str,
    fps: int = 16,
    chunk_sec: float = 5.0,
) -> Path:
    """Wan2.2 S2V로 TTS 오디오에 맞춘 talking-head 비디오를 생성한다."""
    base_image_path = _resolve_results_url(base_image_path)
    audio_path = _resolve_results_url(audio_path)
    if not os.path.exists(base_image_path):
        raise FileNotFoundError(f"base image not found: {base_image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio not found: {audio_path}")

    logger.info("Talking video gen: image=%s audio=%s prompt=%s", base_image_path, audio_path, prompt)

    audio_chunks = await window.chunk_audio_smart(
        audio_path,
        target_range=(5.0, 8.0),
        out_dir=TMP_DIR,
    )
    chunk_paths: list[Path] = []
    current_image = base_image_path

    try:
        for i, audio_chunk in enumerate(audio_chunks):
            image_name = f"s2v_{uuid.uuid4().hex[:6]}_{i}.png"
            audio_name = f"s2v_{uuid.uuid4().hex[:6]}_{i}{audio_chunk.suffix or '.wav'}"
            await comfyui_client.upload_image(current_image, image_name)
            await comfyui_client.upload_audio(str(audio_chunk), audio_name)

            duration = await window.get_media_duration(str(audio_chunk))
            frames = max(9, int(duration * fps) + 1)
            chunk = await runner.run(
                "video.s2v.wan22",
                prompt=prompt,
                image_name=image_name,
                audio_name=audio_name,
                frames=frames,
            )
            chunk_paths.append(chunk)

            if i < len(audio_chunks) - 1:
                current_image = str(await window.extract_last_frame(str(chunk), out_dir=TMP_DIR))

        out = Path(RESULTS_DIR) / f"talking_vid_{uuid.uuid4().hex[:8]}.mp4"
        if len(chunk_paths) == 1:
            shutil.copy(chunk_paths[0], out)
        else:
            await window.concat_videos([str(p) for p in chunk_paths], str(out))
        return await _ensure_audio_duration_matches(out, audio_path)
    finally:
        for audio_chunk in audio_chunks:
            try:
                audio_chunk.unlink()
            except OSError:
                pass


# ─── 비디오 편집 ─────────────────────────────────────────────────────────
async def edit_video(
    video_path: str,
    audio_path: str | None = None,
    image_path: str | None = None,
    prompt: str = "",
) -> Path:
    """오디오 덧입히기 + 이미지 append (기존 동작 유지)."""
    out_path = Path(RESULTS_DIR) / f"edited_{uuid.uuid4().hex[:8]}.mp4"
    video_path = _resolve_results_url(video_path)
    audio_path = _resolve_results_url(audio_path) if audio_path else audio_path
    image_path = _resolve_results_url(image_path) if image_path else image_path
    clip = VideoFileClip(video_path)
    new_audio = None
    try:
        if audio_path and os.path.exists(audio_path):
            new_audio = AudioFileClip(audio_path)
            if "오디오에 맞추기" in prompt:
                from moviepy.video.fx import Loop
                clip = clip.with_effects([Loop(duration=new_audio.duration)])
                clip = clip.with_audio(new_audio)
            else:
                new_audio = new_audio.subclipped(0, min(clip.duration, new_audio.duration))
                clip = clip.with_audio(new_audio)

        if image_path and os.path.exists(image_path):
            img_clip = ImageClip(image_path).with_duration(3.0)
            clip = concatenate_videoclips([clip, img_clip])

        clip.write_videofile(str(out_path), codec="libx264", audio_codec="aac", logger=None)
    finally:
        clip.close()
        if new_audio:
            new_audio.close()
    return out_path


def _resolve_results_url(value: str) -> str:
    prefix = "/api/results/"
    if value.startswith(prefix):
        return os.path.join(RESULTS_DIR, os.path.basename(value))
    return value


def _copy_to_comfy_input(local_path: str, filename: str) -> Path:
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    dest = Path(COMFY_INPUT_DIR) / filename
    shutil.copy(local_path, dest)
    return dest


async def _ensure_audio_duration_matches(video_path: Path, audio_path: str) -> Path:
    """최종 파일에 원본 TTS 오디오가 들어가고 길이가 오디오 기준으로 맞도록 보정."""
    fixed_path = video_path.with_name(f"{video_path.stem}_synced{video_path.suffix}")
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(video_path),
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(fixed_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"sync audio/video failed: {err.decode()}")
    return fixed_path


async def _handle_video_lipsync(inputs: dict[str, Any], ctx: "ExecutionContext") -> dict[str, Any]:
    # Placeholder if called from executor directly
    pass


async def generate_lecture_video(
    face_image_path: str,
    audio_path: str,
    prompt: str = "",
) -> Path:
    """Stage 1: Gen Background + Face Swap -> Stage 2: Idle Loop -> Stage 3: Chunked Lip-sync."""
    import media_image
    from media_engine import gpu_arbiter

    # 영상 생성 전체 과정을 하나의 'heavy' 락으로 감싸서 
    # 작업 중간에 vLLM이 재시작되는 것을 방지 (속도 대폭 향상)
    async with gpu_arbiter.acquire("heavy"):
        # 1. 배경 & 인물 생성 (Stage 1a)
        logger.info("Lecture Stage 1a: Generating base image...")
        lecture_prompt = f"A professional lecturer, medium portrait shot, front view, looking at camera, {prompt}, auditorium background, cinematic lighting, 8k, sharp focus, high detail"
        # Note: media_image.generate_image 내부에서도 acquire를 시도하지만, 
        # gpu_arbiter의 lock은 재진입(reentrant)이 가능하거나 
        # 이미 락을 쥐고 있으면 패스하므로 안전합니다.
        base_image = await media_image.generate_image(lecture_prompt, workflow="flux")
        
        # 2. 페이스 스왑 (Stage 1b)
        logger.info("Lecture Stage 1b: Swapping face...")
        face_image_path = _resolve_results_url(face_image_path)
        swapped_image = await _face_swap(face_image_path, str(base_image))
        
        # 3. 아이들 루프 생성 (Stage 2)
        logger.info("Lecture Stage 2: Generating idle loop...")
        idle_video = await _generate_idle_loop(str(swapped_image))
        
        # 4. 스마트 오디오 분할 및 청크별 립싱크 (Stage 3)
        logger.info("Lecture Stage 3: Chunking audio and lipsyncing...")
        audio_path = _resolve_results_url(audio_path)
        audio_chunks = await window.chunk_audio_smart(
            audio_path,
            target_range=(30.0, 60.0), # 청크 크기를 키워 오버헤드 감소
            out_dir=TMP_DIR,
        )
        
        chunk_paths: list[Path] = []
        try:
            for i, audio_chunk in enumerate(audio_chunks):
                logger.info("Processing lecture chunk %d/%d", i + 1, len(audio_chunks))
                duration = await window.get_media_duration(str(audio_chunk))
                
                # idle_video를 현재 오디오 길이에 맞춰 루핑
                looped_idle = Path(TMP_DIR) / f"idle_loop_{uuid.uuid4().hex[:6]}.mp4"
                cmd = [
                    "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(idle_video),
                    "-t", str(duration), "-c", "copy", str(looped_idle)
                ]
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()
                
                # 립싱크 실행 (LivePortrait 사용 - 고품질)
                chunk_video = await lipsync_video(
                    str(looped_idle), 
                    str(audio_chunk), 
                    workflow="video.lipsync.liveportrait"
                )
                chunk_paths.append(chunk_video)
                looped_idle.unlink(missing_ok=True)

            # 최종 병합
            out = Path(RESULTS_DIR) / f"lecture_final_{uuid.uuid4().hex[:8]}.mp4"
            if len(chunk_paths) == 1:
                shutil.copy(chunk_paths[0], out)
            else:
                await window.concat_videos([str(p) for p in chunk_paths], str(out))
                
            return await _ensure_audio_duration_matches(out, audio_path)
            
        finally:
            for audio_chunk in audio_chunks:
                audio_chunk.unlink(missing_ok=True)


async def _face_swap(source_path: str, target_path: str) -> Path:
    source_name = f"face_src_{uuid.uuid4().hex[:6]}.png"
    target_name = f"face_tgt_{uuid.uuid4().hex[:6]}.png"
    await comfyui_client.upload_image(source_path, source_name)
    await comfyui_client.upload_image(target_path, target_name)
    return await runner.run("image.face_swap.reactor", source_image=source_name, input_image=target_name)


async def _generate_idle_loop(image_path: str) -> Path:
    image_name = f"idle_in_{uuid.uuid4().hex[:6]}.png"
    await comfyui_client.upload_image(image_path, image_name)
    return await runner.run("video.idle_loop.animatediff", image_name=image_name)


async def lipsync_video(
    input_path: str,
    audio_path: str,
    workflow: str = "video.lipsync.wav2lip",
) -> Path:
    """Wav2Lip 또는 LivePortrait를 사용하여 입모양 동기화."""
    input_path = _resolve_results_url(input_path)
    audio_path = _resolve_results_url(audio_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    ext = Path(input_path).suffix.lower()
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv"}
    
    filename = f"lipsync_in_{uuid.uuid4().hex[:6]}{ext}"
    audio_name = f"lipsync_aud_{uuid.uuid4().hex[:6]}{Path(audio_path).suffix or '.wav'}"

    await comfyui_client.upload_image(input_path, filename)
    await comfyui_client.upload_audio(audio_path, audio_name)

    logger.info("Lipsync start: in=%s aud=%s wf=%s", filename, audio_name, workflow)

    params = {
        "audio_name": audio_name,
        "image_name": filename,
        "fps": 25,
    }

    out = await runner.run(workflow, **params)
    return out


# ─── 비디오 단축 (shorts) ────────────────────────────────────────────────
async def shorten_video(video_path: str, prompt: str) -> Path:
    """간단 자르기 + 9:16 세로 크롭."""
    out_path = Path(RESULTS_DIR) / f"shorts_{uuid.uuid4().hex[:8]}.mp4"
    clip = VideoFileClip(video_path)
    try:
        dur = clip.duration
        short_dur = min(30.0, dur / 3)
        start = 0.0
        low = prompt.lower()
        if "끝" in prompt or "end" in low:
            start = max(0.0, dur - short_dur)
        elif "중간" in prompt or "middle" in low:
            start = max(0.0, (dur / 2) - (short_dur / 2))
        sub = clip.subclipped(start, start + short_dur)
        w, h = sub.size
        target_w = int(h * 9 / 16)
        if w > target_w:
            x1 = (w - target_w) // 2
            sub = sub.cropped(x1=x1, y1=0, x2=x1 + target_w, y2=h)
        sub.write_videofile(str(out_path), codec="libx264", audio_codec="aac", logger=None)
    finally:
        clip.close()
    return out_path


# ─── 비디오 분석 (다중 키프레임 VLM) ────────────────────────────────────
async def analyze_video(video_path: str, prompt: str) -> str:
    """3개 키프레임을 VLM에 보내 종합 분석."""
    clip = VideoFileClip(video_path)
    try:
        dur = clip.duration
        times = [0.5, dur / 2, max(0.5, dur - 0.5)]
        frame_b64 = []
        for i, t in enumerate(times):
            p = os.path.join(TMP_DIR, f"vlm_{uuid.uuid4().hex[:4]}_{i}.jpg")
            clip.save_frame(p, t=t)
            with open(p, "rb") as f:
                frame_b64.append(f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}")
            os.unlink(p)
    finally:
        clip.close()

    summaries = []
    for i, b64 in enumerate(frame_b64):
        s = await llm_service.analyze_image(
            b64,
            f"[키프레임 {i+1}/3, t={times[i]:.1f}s] {prompt}",
        )
        summaries.append(s)
    combined = "\n\n".join(f"[프레임 {i+1}] {s}" for i, s in enumerate(summaries))
    return await llm_service.generate_text(
        f"다음 3개 키프레임 분석을 종합해 비디오를 설명하세요.\n{combined}\n\n사용자 질문: {prompt}",
        "You are a helpful video analyst.",
    )
