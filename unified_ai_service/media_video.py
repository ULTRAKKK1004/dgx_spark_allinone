"""비디오 생성·편집·분석 (media_engine 사용하는 얇은 shim).

generate_long_video: Wan2.2 i2v로 moving-window 방식 긴 비디오 생성.
edit_video: moviepy 기반 (오디오 덧입히기 / 이미지 append).
shorten_video: 기본 자르기 + 세로 크롭.
analyze_video: 다중 키프레임을 VLM에 전달.
"""
import base64
import logging
import os
import uuid
from pathlib import Path

from moviepy import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips

import llm_service
from media_engine import runner, comfyui_client, window

logger = logging.getLogger(__name__)

BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TMP_DIR = os.path.join(BASE_DIR, "tmp_video")
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


# ─── 비디오 편집 ─────────────────────────────────────────────────────────
async def edit_video(
    video_path: str,
    audio_path: str | None = None,
    image_path: str | None = None,
    prompt: str = "",
) -> Path:
    """오디오 덧입히기 + 이미지 append (기존 동작 유지)."""
    out_path = Path(RESULTS_DIR) / f"edited_{uuid.uuid4().hex[:8]}.mp4"
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
