"""Natural-language multimodal planner."""
from __future__ import annotations

import json
import os
import re
import asyncio
from typing import Any

import llm_service
from media_capabilities import SUPPORTED_ACTIONS, planner_prompt
from multimodal_models import MediaAsset, MediaPlan


PLANNER_SYSTEM_PROMPT = "You convert user media requests into executable JSON plans."


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


async def plan_request(
    instruction: str,
    assets: list[MediaAsset],
    quality: str = "standard",
    preferred_voice_provider: str = "auto",
    planner_timeout_sec: float | None = None,
) -> MediaPlan:
    try:
        timeout = planner_timeout_sec
        if timeout is None:
            timeout = float(os.getenv("MULTIMODAL_PLANNER_TIMEOUT", "8"))
        raw = await asyncio.wait_for(
            _llm_plan(instruction, assets, quality, preferred_voice_provider),
            timeout=timeout,
        )
        return MediaPlan.from_dict(
            raw,
            supported_actions=SUPPORTED_ACTIONS,
            asset_aliases={asset.alias for asset in assets},
        )
    except Exception:
        return fallback_plan(instruction, assets, quality, preferred_voice_provider)


async def _llm_plan(
    instruction: str,
    assets: list[MediaAsset],
    quality: str,
    preferred_voice_provider: str,
) -> dict[str, Any]:
    asset_lines = [
        f"- {asset.alias}: filename={asset.filename or asset.path}, mime={asset.mime_type}"
        for asset in assets
    ]
    prompt = "\n".join(
        [
            planner_prompt(),
            "",
            f"Quality: {quality}",
            f"Preferred voice provider: {preferred_voice_provider}",
            "Uploaded assets:",
            "\n".join(asset_lines) if asset_lines else "- none",
            "",
            f"User instruction: {instruction}",
        ]
    )
    response = await llm_service.generate_text(prompt, PLANNER_SYSTEM_PROMPT)
    return json.loads(_strip_json_fence(response))


def fallback_plan(
    instruction: str,
    assets: list[MediaAsset],
    quality: str = "standard",
    preferred_voice_provider: str = "auto",
) -> MediaPlan:
    low = instruction.lower()
    image_assets = [a for a in assets if a.mime_type.startswith("image/")]
    audio_assets = [a for a in assets if a.mime_type.startswith("audio/")]
    video_assets = [a for a in assets if a.mime_type.startswith("video/")]

    if _has_any(low, ["ppt", "발표자료", "슬라이드", "프레젠테이션"]):
        raw = _single("ppt.generate", {"topic": instruction}, "ppt", "deck", "document", quality)
    elif video_assets and _has_any(low, ["분석", "설명", "analyze"]):
        raw = _single("video.analyze", {"video": video_assets[0].alias, "prompt": instruction}, "text", "analysis", "text", quality)
    elif image_assets and _has_any(low, ["분석", "설명", "analyze"]):
        raw = _single("image.analyze", {"image": image_assets[0].alias, "prompt": instruction}, "text", "analysis", "text", quality)
    elif audio_assets and _has_any(low, ["전사", "텍스트", "자막", "transcribe", "stt"]):
        raw = _single("audio.transcribe", {"audio": audio_assets[0].alias}, "text", "transcript", "text", quality)
    elif _has_any(low, ["음성", "내레이션", "나레이션", "tts", "읽어", "말해"]):
        raw = _single(
            "voice.tts",
            {"text": instruction, "provider": preferred_voice_provider},
            "audio",
            "voice",
            "audio",
            quality,
        )
    elif image_assets and _has_any(low, ["영상", "비디오", "움직", "video"]):
        raw = _single("video.generate", {"image": image_assets[0].alias, "prompt": instruction}, "video", "video", "video", quality)
    elif image_assets and len(image_assets) >= 2 and _has_any(low, ["마스크", "제거", "inpaint"]):
        raw = _single(
            "image.inpaint",
            {"image": image_assets[0].alias, "mask": image_assets[1].alias, "prompt": instruction},
            "image",
            "image",
            "image",
            quality,
        )
    elif image_assets and _has_any(low, ["편집", "수정", "바꿔", "제거", "edit"]):
        raw = _single("image.edit", {"image": image_assets[0].alias, "prompt": instruction}, "image", "image", "image", quality)
    elif _has_any(low, ["음악", "배경음", "music", "bgm"]):
        raw = _single("audio.music", {"prompt": instruction, "duration": _extract_duration(low, 30)}, "audio", "music", "audio", quality)
    elif _has_any(low, ["이미지", "그림", "사진", "image"]):
        raw = _single("image.generate", {"prompt": instruction}, "image", "image", "image", quality)
    else:
        raw = _single("text.generate", {"prompt": instruction}, "text", "answer", "text", quality)

    return MediaPlan.from_dict(raw, supported_actions=SUPPORTED_ACTIONS, asset_aliases={asset.alias for asset in assets})


def _single(action: str, inputs: dict[str, Any], output_key: str, alias: str, final_format: str, quality: str) -> dict[str, Any]:
    return {
        "version": "1",
        "goal": action,
        "quality": quality,
        "steps": [{"id": "step_1", "action": action, "inputs": inputs, "outputs": {output_key: alias}}],
        "final": {"primary": alias, "format": final_format},
    }


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _extract_duration(text: str, default: int) -> int:
    match = re.search(r"(\d+)\s*(초|sec|seconds?)", text)
    if match:
        return max(1, int(match.group(1)))
    match = re.search(r"(\d+)\s*(분|min|minutes?)", text)
    if match:
        return max(1, int(match.group(1)) * 60)
    return default
