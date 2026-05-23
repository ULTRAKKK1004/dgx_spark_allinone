"""Sequential executor for validated MediaPlan objects."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from media_capabilities import SUPPORTED_ACTIONS
from multimodal_models import MediaAsset, MediaPlan


BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class StepExecutionError(RuntimeError):
    """Raised when a single plan step fails."""


class ExecutionContext:
    def __init__(self, assets: list[MediaAsset], quality: str):
        self.assets: dict[str, MediaAsset] = {asset.alias: asset for asset in assets}
        self.values: dict[str, Any] = {}
        self.quality = quality

    def resolve(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            alias = value[1:]
            return self.values.get(alias, value)
        if isinstance(value, str) and value in self.assets:
            return self.assets[value].path
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        return value

    def resolve_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: self.resolve(value) for key, value in inputs.items()}


Handler = Callable[[dict[str, Any], ExecutionContext], Awaitable[dict[str, Any]]]


async def execute_plan(plan: MediaPlan, assets: list[MediaAsset]) -> dict[str, Any]:
    ctx = ExecutionContext(assets, quality=plan.quality)
    step_results: list[dict[str, Any]] = []
    for step in plan.steps:
        try:
            handler = ACTION_HANDLERS[step.action]
            resolved_inputs = ctx.resolve_inputs(step.inputs)
            raw_result = await handler(resolved_inputs, ctx)
            normalized = _normalize_result(raw_result)
            for output_key, alias in step.outputs.items():
                if output_key in normalized:
                    ctx.values[alias] = normalized[output_key]
                elif len(normalized) == 1:
                    ctx.values[alias] = next(iter(normalized.values()))
                else:
                    raise StepExecutionError(f"{step.id}: output {output_key} missing")
            step_results.append(
                {
                    "id": step.id,
                    "action": step.action,
                    "status": "completed",
                    "result": normalized,
                }
            )
        except Exception as exc:
            step_results.append(
                {
                    "id": step.id,
                    "action": step.action,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if step.optional:
                continue
            raise StepExecutionError(f"{step.id} ({step.action}) failed: {exc}") from exc

    primary = plan.final.get("primary")
    primary_value = ctx.values.get(primary) if primary else None
    final = {"type": plan.final.get("format", "unknown"), "primary": primary}
    if isinstance(primary_value, str) and primary_value.startswith("/api/results/"):
        final["url"] = primary_value
    else:
        final["value"] = primary_value
    return {"plan": plan.to_dict(), "steps": step_results, "final": final}


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, Path):
            normalized[key] = _path_to_url_or_str(str(value))
        elif isinstance(value, str):
            normalized[key] = _path_to_url_or_str(value)
        else:
            normalized[key] = value
    return normalized


def _path_to_url_or_str(value: str) -> str:
    abs_results = os.path.abspath(RESULTS_DIR)
    try:
        abs_value = os.path.abspath(value)
    except OSError:
        return value
    if abs_value.startswith(abs_results + os.sep):
        return f"/api/results/{os.path.basename(abs_value)}"
    return value


async def _handle_text_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import llm_service

    text = await _await_llm_step(
        llm_service.generate_text(
            inputs.get("prompt", ""),
            inputs.get("system_prompt", "You are a helpful multimodal media assistant."),
        ),
        "text.generate",
    )
    return {"text": text}


async def _handle_ppt_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import llm_service
    import ppt_service

    topic = inputs.get("topic", "")
    slides = await _await_llm_step(llm_service.generate_ppt_structure(topic), "ppt.generate")
    output_path = os.path.join(RESULTS_DIR, f"presentation_{uuid.uuid4().hex[:8]}.pptx")
    ppt_service.generate_ppt_file(topic, slides, output_path)
    return {"ppt": output_path}


async def _handle_image_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_image

    path = await media_image.generate_image(
        inputs.get("prompt", ""),
        workflow=inputs.get("workflow", "zimage_turbo"),
    )
    return {"image": path}


async def _handle_image_edit(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_image

    path = await media_image.edit_image(inputs["image"], inputs.get("prompt", ""))
    return {"image": path}


async def _handle_image_control(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_image

    path = await media_image.control_image(
        inputs.get("prompt", ""),
        inputs["control_image"],
        control_type=inputs.get("control_type", "canny"),
        strength=float(inputs.get("strength", 0.7)),
    )
    return {"image": path}


async def _handle_image_inpaint(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_image

    path = await media_image.inpaint_image(inputs["image"], inputs["mask"], inputs.get("prompt", ""))
    return {"image": path}


async def _handle_image_analyze(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import llm_service

    data_url = _file_to_data_url(inputs["image"], fallback_mime="image/png")
    text = await _await_llm_step(
        llm_service.analyze_image(data_url, inputs.get("prompt", "Analyze this image.")),
        "image.analyze",
    )
    return {"text": text}


async def _handle_audio_music(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_audio

    duration = int(inputs.get("duration", 10))
    if duration > 30:
        path = await media_audio.generate_long_music(inputs.get("prompt", ""), duration)
    else:
        path = await media_audio.generate_music(inputs.get("prompt", ""), duration)
    return {"audio": path}


async def _handle_audio_transcribe(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import stt_service

    return {"text": stt_service.transcribe_audio(inputs["audio"])}


async def _handle_voice_tts(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_audio

    result = await media_audio.synthesize_voice(
        inputs.get("text", ""),
        provider=inputs.get("provider", "auto"),
        quality=ctx.quality,
        voice=inputs.get("voice", "default"),
        ref_audio=inputs.get("ref_audio", ""),
        ref_text=inputs.get("ref_text", ""),
    )
    return {"audio": result["url"], "metadata": result}


async def _handle_video_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    path = await media_video.generate_long_video(
        inputs.get("prompt", ""),
        inputs["image"],
        total_duration_target=int(inputs.get("duration", 30)),
    )
    return {"video": path}


async def _handle_video_edit(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    path = await media_video.edit_video(
        inputs["video"],
        audio_path=inputs.get("audio"),
        image_path=inputs.get("image"),
        prompt=inputs.get("prompt", ""),
    )
    return {"video": path}


async def _handle_video_analyze(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    return {"text": await media_video.analyze_video(inputs["video"], inputs.get("prompt", ""))}


async def _handle_video_shorts(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    return {"video": await media_video.shorten_video(inputs["video"], inputs.get("prompt", ""))}


async def _handle_video_lipsync(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    raise RuntimeError("video.lipsync is registered but requires Phase B4 lip-sync engine installation")


async def _handle_package_bundle(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    items = inputs.get("items", [])
    manifest = {"items": [ctx.resolve(item) for item in items]}
    output_path = os.path.join(RESULTS_DIR, f"bundle_{uuid.uuid4().hex[:8]}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {"manifest": output_path}


def _file_to_data_url(path: str, fallback_mime: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{fallback_mime};base64,{encoded}"


async def _await_llm_step(awaitable, label: str):
    timeout = float(os.getenv("MULTIMODAL_LLM_STEP_TIMEOUT", "20"))
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} exceeded {timeout:g}s") from exc


ACTION_HANDLERS: dict[str, Handler] = {
    "text.generate": _handle_text_generate,
    "ppt.generate": _handle_ppt_generate,
    "image.generate": _handle_image_generate,
    "image.edit": _handle_image_edit,
    "image.control": _handle_image_control,
    "image.inpaint": _handle_image_inpaint,
    "image.analyze": _handle_image_analyze,
    "audio.music": _handle_audio_music,
    "audio.transcribe": _handle_audio_transcribe,
    "voice.tts": _handle_voice_tts,
    "video.generate": _handle_video_generate,
    "video.edit": _handle_video_edit,
    "video.analyze": _handle_video_analyze,
    "video.shorts": _handle_video_shorts,
    "video.lipsync": _handle_video_lipsync,
    "package.bundle": _handle_package_bundle,
}
