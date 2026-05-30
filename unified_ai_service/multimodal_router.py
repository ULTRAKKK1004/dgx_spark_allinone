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
    preferred_voice: str = "default",
    planner_timeout_sec: float | None = None,
) -> MediaPlan:
    # 1. Try rule-based fallback first for better keyword reliability (music, image, etc.)
    f_plan = fallback_plan(instruction, assets, quality, preferred_voice_provider, preferred_voice)
    
    # If it matched a specific media action (not just generic text), use it immediately
    if f_plan.goal != "text.generate":
        return f_plan

    # 2. Try LLM-based planning for complex/ambiguous instructions
    try:
        timeout = planner_timeout_sec
        if timeout is None:
            timeout = float(os.getenv("MULTIMODAL_PLANNER_TIMEOUT", "10"))
            
        raw = await asyncio.wait_for(
            _llm_plan(instruction, assets, quality, preferred_voice_provider, preferred_voice),
            timeout=timeout,
        )
        plan = MediaPlan.from_dict(
            raw,
            supported_actions=SUPPORTED_ACTIONS,
            asset_aliases={asset.alias for asset in assets},
        )
        # If the LLM plan is generic text, but the instruction is non-empty,
        # we still prefer it over f_plan if it has more steps.
        return plan
    except Exception:
        # Final fallback to generic text generate
        return f_plan


async def _llm_plan(
    instruction: str,
    assets: list[MediaAsset],
    quality: str,
    preferred_voice_provider: str,
    preferred_voice: str,
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
            f"Preferred ElevenLabs voice id: {preferred_voice}",
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
    preferred_voice: str = "default",
) -> MediaPlan:
    low = instruction.lower()
    image_assets = [a for a in assets if a.mime_type.startswith("image/")]
    audio_assets = [a for a in assets if a.mime_type.startswith("audio/")]
    video_assets = [a for a in assets if a.mime_type.startswith("video/")]
    text_assets = [a for a in assets if _is_text_asset(a)]

    if _is_voiced_lecture_video_request(low, image_assets, text_assets):
        if _has_any(low, ["긴", "30분", "ppt", "피피티", "프로", "long", "presentation"]):
            raw = {
                "version": "1",
                "goal": "video.lecture.pro",
                "quality": quality,
                "steps": [
                    {"id": "step_1", "action": "text.extract", "inputs": {"file": text_assets[0].alias}, "outputs": {"text": "script_text"}},
                    {"id": "step_2", "action": "video.lecture.pro", "inputs": {"image": image_assets[0].alias, "script": "$script_text", "topic": instruction}, "outputs": {"video": "pro_lecture"}}
                ],
                "final": {"primary": "pro_lecture", "format": "video"}
            }
        else:
            raw = _voiced_lecture_video_plan(
                instruction,
                image_assets[0].alias,
                text_assets[0].alias,
                quality,
                preferred_voice_provider,
                preferred_voice,
            )
    elif _has_any(low, ["드라마", "광고", "애니메이션", "drama", "animation", "ad", "commercial", "storyboard"]):
        raw = _single("video.storyboard", {"prompt": instruction, "genre": "animation" if "애니" in low else "cinematic"}, "video", "story_video", "video", quality)
    elif _has_any(low, ["ppt", "발표자료", "슬라이드", "프레젠테이션", "presentation", "deck", "slides"]):
        ppt_inputs = {"topic": instruction}
        # Check if an HTML or PPTX file was uploaded to use as template/source
        doc_assets = [a for a in assets if a.filename and a.filename.lower().endswith((".pptx", ".html", ".htm"))]
        if doc_assets:
            ppt_inputs["template"] = doc_assets[0].alias
        raw = _single("ppt.generate", ppt_inputs, "ppt", "deck", "document", quality)
    elif video_assets and _has_any(low, ["분석", "설명", "analyze", "explain"]):
        raw = _single("video.analyze", {"video": video_assets[0].alias, "prompt": instruction}, "text", "analysis", "text", quality)
    elif image_assets and _has_any(low, ["분석", "설명", "analyze", "explain"]):
        raw = _single("image.analyze", {"image": image_assets[0].alias, "prompt": instruction}, "text", "analysis", "text", quality)
    elif audio_assets and _has_any(low, ["전사", "텍스트", "transcribe", "stt", "text"]):
        raw = _single("audio.transcribe", {"audio": audio_assets[0].alias}, "text", "transcript", "text", quality)
    elif audio_assets and _has_any(low, ["자막", "srt", "subtitle"]):
        raw = _single("audio.subtitle", {"audio": audio_assets[0].alias}, "srt", "subtitle", "document", quality)
    elif _has_any(low, ["음성", "내레이션", "나레이션", "tts", "읽어", "말해", "say", "speak", "voice", "narrate"]):
        raw = _single(
            "voice.tts",
            {"text": instruction, "provider": preferred_voice_provider, "voice": preferred_voice},
            "audio",
            "voice",
            "audio",
            quality,
        )
    elif image_assets and _has_any(low, ["영상", "비디오", "움직", "video", "animate", "motion"]):
        raw = _single("video.generate", {"image": image_assets[0].alias, "prompt": instruction}, "video", "video", "video", quality)
    elif image_assets and len(image_assets) >= 2 and _has_any(low, ["마스크", "제거", "inpaint", "mask", "remove"]):
        raw = _single(
            "image.inpaint",
            {"image": image_assets[0].alias, "mask": image_assets[1].alias, "prompt": instruction},
            "image",
            "image",
            "image",
            quality,
        )
    elif image_assets and _has_any(low, ["편집", "수정", "바꿔", "제거", "edit", "modify", "change"]):
        raw = _single("image.edit", {"image": image_assets[0].alias, "prompt": instruction}, "image", "image", "image", quality)
    elif _has_any(low, ["음악", "배경음", "music", "bgm", "song", "track", "melody", "rhythm", "beat", "instrumental", "concerto", "symphony", "piano", "guitar", "violin"]):
        raw = _single("audio.music", {"prompt": instruction, "duration": _extract_duration(low, 30)}, "audio", "music", "audio", quality)
    elif _has_any(low, ["이미지", "그림", "사진", "image", "draw", "paint", "picture", "photo", "portrait", "sketch"]):
        if quality == "draft":
            raw = _single("image.generate", {"prompt": instruction, "workflow": "flux", "workflow_type": "schnell", "steps": 4}, "image", "image", "image", quality)
        else:
            # Standard/High: Use FLUX Dev for best prompt adherence
            raw = _single("image.generate", {"prompt": instruction, "workflow": "flux", "workflow_type": "dev"}, "image", "image", "image", quality)
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


def _voiced_lecture_video_plan(
    instruction: str,
    image_alias: str,
    script_alias: str,
    quality: str,
    preferred_voice_provider: str,
    preferred_voice: str,
) -> dict[str, Any]:
    tts_provider = _voice_provider_for_lecture(preferred_voice_provider, preferred_voice)
    return {
        "version": "1",
        "goal": "lecture.video",
        "quality": quality,
        "steps": [
            {
                "id": "step_1",
                "action": "text.extract",
                "inputs": {"file": script_alias},
                "outputs": {"text": "script_text"},
            },
            {
                "id": "step_2",
                "action": "voice.tts",
                "inputs": {
                    "text": "$script_text",
                    "provider": tts_provider,
                    "voice": preferred_voice,
                },
                "outputs": {"audio": "voice_audio"},
            },
            {
                "id": "step_3",
                "action": "video.lecture",
                "inputs": {
                    "image": image_alias,
                    "audio": "$voice_audio",
                    "prompt": instruction,
                },
                "outputs": {"video": "lecture_video"},
            },
            {
                "id": "step_4",
                "action": "audio.subtitle",
                "inputs": {"audio": "$voice_audio"},
                "outputs": {"srt": "lecture_subtitle"},
            },
        ],
        "final": {"primary": "lecture_video", "format": "video", "secondary": "lecture_subtitle"},
    }


def _voice_provider_for_lecture(preferred_voice_provider: str, preferred_voice: str) -> str:
    return preferred_voice_provider


def _is_voiced_lecture_video_request(
    low_instruction: str,
    image_assets: list[MediaAsset],
    text_assets: list[MediaAsset],
) -> bool:
    if not image_assets or not text_assets:
        return False
    return _has_any(low_instruction, ["강의", "lecture", "발표", "present", "강사", "입모양", "립싱크"]) and _has_any(
        low_instruction,
        ["대본", "script", "영상", "비디오", "video", "말하", "읽어", "tts", "싱크", "음성"],
    )


def _is_text_asset(asset: MediaAsset) -> bool:
    if asset.mime_type.startswith("text/"):
        return True
    name = (asset.filename or asset.path).lower()
    return name.endswith((".txt", ".md", ".markdown", ".srt", ".vtt", ".csv", ".json"))


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
