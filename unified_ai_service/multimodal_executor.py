"""Sequential executor for validated MediaPlan objects."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import logging
from media_capabilities import SUPPORTED_ACTIONS
from multimodal_models import MediaAsset, MediaPlan

logger = logging.getLogger(__name__)

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
    logger.info("Executing plan for goal: %s (quality=%s)", plan.goal, plan.quality)
    for step in plan.steps:
        try:
            logger.info("Step %s: action=%s inputs=%s", step.id, step.action, step.inputs)
            handler = ACTION_HANDLERS[step.action]
            resolved_inputs = ctx.resolve_inputs(step.inputs)
            raw_result = await handler(resolved_inputs, ctx)
            normalized = _normalize_result(raw_result)
            for output_key, alias in step.outputs.items():
                if output_key in raw_result:
                    ctx.values[alias] = _value_for_chaining(raw_result, output_key)
                elif len(raw_result) == 1:
                    ctx.values[alias] = _value_for_chaining(raw_result)
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
    normalized_primary = _normalize_result({"primary": primary_value}).get("primary")
    final = {"type": plan.final.get("format", "unknown"), "primary": primary}
    if isinstance(normalized_primary, str) and normalized_primary.startswith("/api/results/"):
        final["url"] = normalized_primary
    else:
        final["value"] = normalized_primary
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


def _value_for_chaining(result: dict[str, Any], output_key: str | None = None) -> Any:
    value = result[output_key] if output_key else next(iter(result.values()))
    if isinstance(value, str) and value.startswith("/api/results/"):
        metadata = result.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            return metadata["path"]
        return _results_url_to_path(value)
    if isinstance(value, Path):
        return str(value)
    return value


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


async def _handle_text_extract(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = inputs.get("file")
    if not isinstance(path, str) or not path:
        raise ValueError("text.extract requires a file input")
    return {"text": _read_text_file(path)}


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

    params = inputs.copy()
    workflow = params.pop("workflow", "zimage_turbo")
    if workflow == "flux" and "workflow_type" in params:
        params["workflow"] = params.pop("workflow_type")

    path = await media_image.generate_image(
        inputs.get("prompt", ""),
        workflow=workflow,
        **params,
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
    import media_image
    import llm_service

    image_path = inputs["image"]
    prompt = inputs.get("prompt", "Analyze this image.")

    if ctx.quality == "high":
        # Janus-Pro (Local GPU, higher quality)
        # Note: media_image.analyze_image_janus already handles the ComfyUI logic
        text = await media_image.analyze_image_janus(image_path, prompt)
    else:
        # VLM via LLM Service (Fast, lower/standard quality)
        data_url = _file_to_data_url(image_path, fallback_mime="image/png")
        text = await _await_llm_step(
            llm_service.analyze_image(data_url, prompt),
            "image.analyze",
        )

    return {"text": text}


async def _handle_audio_music(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_audio

    duration = int(inputs.get("duration", 10))
    if duration > 30:
        path = await media_audio.generate_long_music(inputs.get("prompt", ""), duration)
    else:
        path, _ = await media_audio.generate_music(inputs.get("prompt", ""), duration)
    return {"audio": path}


async def _handle_audio_transcribe(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import stt_service

    text = await stt_service.transcribe_audio(
        inputs["audio"],
        language=inputs.get("language"),
    )
    return {"text": text}


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


async def _handle_video_talking(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    path = await media_video.generate_talking_video(
        inputs.get("prompt", ""),
        inputs["image"],
        inputs["audio"],
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
    import media_video

    path = await media_video.lipsync_video(
        inputs["video"] if "video" in inputs else inputs["image"],
        inputs["audio"],
        workflow=inputs.get("workflow", "video.lipsync.wav2lip"),
    )
    return {"video": path}


async def _handle_video_lecture(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import media_video

    path = await media_video.generate_lecture_video(
        inputs["image"],
        inputs["audio"],
        prompt=inputs.get("prompt", ""),
    )
    return {"video": path}


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


def _read_text_file(path: str) -> str:
    local_path = _results_url_to_path(path)
    ext = Path(local_path).suffix.lower()
    if ext == ".pdf":
        return _read_pdf_text(local_path)
    if ext not in {"", ".txt", ".md", ".markdown", ".srt", ".vtt", ".csv", ".json"}:
        raise ValueError(f"text.extract does not support {ext or 'this file type'}")

    max_bytes = int(os.getenv("MULTIMODAL_TEXT_EXTRACT_MAX_BYTES", str(5 * 1024 * 1024)))
    with open(local_path, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"text.extract input exceeds {max_bytes} bytes")
    for encoding in ("utf-8-sig", "utf-8", "cp949", "utf-16"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").strip()


def _read_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("text.extract PDF support requires pypdf") from exc

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _results_url_to_path(value: str) -> str:
    prefix = "/api/results/"
    if value.startswith(prefix):
        return os.path.join(RESULTS_DIR, os.path.basename(value))
    return value


async def _await_llm_step(awaitable, label: str):
    timeout = float(os.getenv("MULTIMODAL_LLM_STEP_TIMEOUT", "20"))
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} exceeded {timeout:g}s") from exc


async def _handle_audio_subtitle(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    import stt_service
    import os, uuid

    audio_path = inputs["audio"]
    language = inputs.get("language")

    segments = await stt_service.transcribe_with_timestamps(audio_path, language=language)
    srt_content = stt_service.generate_srt_from_segments(segments)

    filename = f"sub_{uuid.uuid4().hex[:8]}.srt"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return {"srt": path}


ACTION_HANDLERS: dict[str, Any] = {
    "text.generate": _handle_text_generate,
    "text.extract": _handle_text_extract,
    "ppt.generate": _handle_ppt_generate,
    "image.generate": _handle_image_generate,
    "image.edit": _handle_image_edit,
    "image.control": _handle_image_control,
    "image.inpaint": _handle_image_inpaint,
    "image.analyze": _handle_image_analyze,
    "audio.music": _handle_audio_music,
    "audio.transcribe": _handle_audio_transcribe,
    "audio.subtitle": _handle_audio_subtitle,
    "voice.tts": _handle_voice_tts,
    "video.generate": _handle_video_generate,

    "video.talking": _handle_video_talking,
    "video.edit": _handle_video_edit,
    "video.analyze": _handle_video_analyze,
    "video.shorts": _handle_video_shorts,
    "video.lipsync": _handle_video_lipsync,
    "video.lecture": _handle_video_lecture,
    "package.bundle": _handle_package_bundle,
}
