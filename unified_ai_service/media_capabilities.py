"""Supported multimodal actions and planner-facing capability prompt."""
from __future__ import annotations

from typing import Any


CAPABILITIES: dict[str, dict[str, Any]] = {
    "text.generate": {
        "kind": "text",
        "description": "Generate or rewrite text with the local LLM.",
        "inputs": {"prompt": "string", "system_prompt": "optional string"},
        "outputs": {"text": "string"},
    },
    "text.extract": {
        "kind": "text",
        "description": "Extract script text from an uploaded text/markdown/subtitle file.",
        "inputs": {"file": "text asset alias"},
        "outputs": {"text": "string"},
    },
    "ppt.generate": {
        "kind": "document",
        "description": "Create a high-fidelity 16:9 PowerPoint deck via HTML rendering. Supports complex layouts, charts, and tables.",
        "inputs": {"topic": "string", "instruction": "optional layout instruction"},
        "outputs": {"ppt": "pptx file url"},
    },
    "image.generate": {
        "kind": "image",
        "description": "Generate an image from text. Workflows: zimage_turbo, flux.",
        "inputs": {"prompt": "string", "workflow": "optional string"},
        "outputs": {"image": "image file url"},
    },
    "image.edit": {
        "kind": "image",
        "description": "Edit an uploaded image with an instruction.",
        "inputs": {"image": "asset alias or prior output", "prompt": "string"},
        "outputs": {"image": "image file url"},
    },
    "image.control": {
        "kind": "image",
        "description": "Generate image using FLUX ControlNet with canny/openpose/depth/scribble.",
        "inputs": {
            "control_image": "asset alias",
            "prompt": "string",
            "control_type": "optional string",
            "strength": "optional float",
        },
        "outputs": {"image": "image file url"},
    },
    "image.inpaint": {
        "kind": "image",
        "description": "Inpaint or remove a masked region from an uploaded image.",
        "inputs": {"image": "asset alias", "mask": "asset alias", "prompt": "string"},
        "outputs": {"image": "image file url"},
    },
    "image.analyze": {
        "kind": "image",
        "description": "Analyze an uploaded image with the VLM.",
        "inputs": {"image": "asset alias or prior output", "prompt": "string"},
        "outputs": {"text": "string"},
    },
    "audio.music": {
        "kind": "audio",
        "description": "Generate short or long background music.",
        "inputs": {"prompt": "string", "duration": "optional integer seconds"},
        "outputs": {"audio": "audio file url"},
    },
    "audio.transcribe": {
        "kind": "audio",
        "description": "Transcribe uploaded audio using local Whisper.",
        "inputs": {"audio": "asset alias", "language": "optional string"},
        "outputs": {"text": "string"},
    },
    "audio.subtitle": {
        "kind": "audio",
        "description": "Generate SRT subtitles from uploaded audio.",
        "inputs": {"audio": "asset alias", "language": "optional string"},
        "outputs": {"srt": "srt file content or url"},
    },
    "voice.tts": {
        "kind": "voice",
        "description": "Generate spoken narration from text using local F5-TTS or ElevenLabs when configured.",
        "inputs": {
            "text": "string",
            "provider": "optional auto|local_f5|elevenlabs",
            "voice": "optional string",
        },
        "outputs": {"audio": "audio file url"},
    },
    "video.generate": {
        "kind": "video",
        "description": "Generate long i2v video from an uploaded base image using Wan2.2 moving window.",
        "inputs": {"image": "asset alias", "prompt": "string", "duration": "optional integer seconds"},
        "outputs": {"video": "video file url"},
    },
    "video.edit": {
        "kind": "video",
        "description": "Edit uploaded video with optional audio overlay.",
        "inputs": {"video": "asset alias", "audio": "optional asset alias", "prompt": "optional string"},
        "outputs": {"video": "video file url"},
    },
    "video.analyze": {
        "kind": "video",
        "description": "Analyze a video with sampled keyframes and the VLM.",
        "inputs": {"video": "asset alias", "prompt": "string"},
        "outputs": {"text": "string"},
    },
    "video.shorts": {
        "kind": "video",
        "description": "Create a short vertical clip from an uploaded video.",
        "inputs": {"video": "asset alias", "prompt": "optional string"},
        "outputs": {"video": "video file url"},
    },
    "video.lipsync": {
        "kind": "video",
        "description": "Lip-sync a presenter video or still image to narration using Wav2Lip/LivePortrait (Phase B4).",
        "inputs": {"video": "asset alias or image alias", "audio": "asset alias", "workflow": "optional string"},
        "outputs": {"video": "video file url"},
    },
    "video.lecture": {
        "kind": "video",
        "description": "Generate a full lecture video: Stage 1 (Face Swap) -> Stage 2 (Idle Loop) -> Stage 3 (Lip-sync).",
        "inputs": {"image": "lecturer face alias", "audio": "narration alias", "prompt": "optional lecture topic"},
        "outputs": {"video": "video file url"},
    },
    "video.lecture.pro": {
        "kind": "video",
        "description": "Generate a complex, 30+ min PPT-synced lecture video with PiP and BGM.",
        "inputs": {"image": "lecturer face alias", "script": "script text alias", "topic": "presentation topic"},
        "outputs": {"video": "video file url"},
    },
    "video.storyboard": {
        "kind": "video",
        "description": "Generate multi-scene videos for animation, drama, and ads from a prompt.",
        "inputs": {"prompt": "story/ad description", "genre": "optional genre string"},
        "outputs": {"video": "video file url"},
    },
    "package.bundle": {
        "kind": "package",
        "description": "Bundle multiple prior outputs into a manifest result.",
        "inputs": {"items": "list of aliases"},
        "outputs": {"manifest": "json object"},
    },
}

SUPPORTED_ACTIONS = set(CAPABILITIES)


def get_action(action: str) -> dict[str, Any]:
    return CAPABILITIES[action]


def planner_prompt() -> str:
    lines = [
        "You are a multimodal media planner.",
        "Return JSON only. Do not use markdown fences.",
        "Unsupported actions are invalid.",
        "Instructions:",
        "- Use 'video.lecture' for any task involving a person talking based on a script/audio.",
        "- Do NOT use 'video.talking' as it is unavailable.",
        "- Use 'image.analyze' (or 'image.analyze.janus' if high quality is needed) for image understanding.",
        "- Use 'audio.subtitle' for generating SRT files.",
        "Schema: {version:'1', goal:string, quality:'draft|standard|high', steps:[{id, action, inputs, outputs}], final:{primary, format}}",
        "Available actions:",
    ]
    for name in sorted(CAPABILITIES):
        cap = CAPABILITIES[name]
        lines.append(f"- {name}: {cap['description']} Inputs={cap['inputs']} Outputs={cap['outputs']}")
    return "\n".join(lines)
