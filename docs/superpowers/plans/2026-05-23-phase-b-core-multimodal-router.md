# Phase B Core Multimodal Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `/api/multimodal/execute`, a natural-language multimodal router that creates a validated media plan and executes existing image, audio, voice, video, PPT, and analysis tools through one job pipeline.

**Architecture:** Add a small planning layer (`multimodal_models.py`, `media_capabilities.py`, `multimodal_router.py`) and an execution layer (`multimodal_executor.py`, `voice_providers.py`) on top of the existing media modules. The router uses vLLM JSON planning when available and deterministic rule fallback when unavailable or invalid; executor runs steps sequentially and records every step result in the existing job system.

**Tech Stack:** Python 3.12, FastAPI, pytest, httpx, pydub, existing `media_engine`, existing `llm_service`, existing `job_manager`.

**Spec:** `/home/yanus/docs/superpowers/specs/2026-05-23-phase-b-core-multimodal-router-design.md`

---

## Shared Context

- Work root: `/home/yanus`
- Service root: `/home/yanus/unified_ai_service`
- Python: `/home/yanus/unified_ai_service/venv/bin/python`
- Unit test command: `cd /home/yanus/unified_ai_service && ./venv/bin/python -m pytest tests/ media_engine/tests/ -v`
- Existing result URL convention: local files in `/home/yanus/unified_ai_service/results/` are returned as `/api/results/<filename>`
- Existing uploads directory: `/home/yanus/unified_ai_service/uploads/`
- Commit prefix for this plan: `feat(multimodal):`, `test(multimodal):`, `docs(multimodal):`
- Do not run GPU-heavy integration by default.
- Do not require `ELEVENLABS_API_KEY` for tests or local operation.

---

## File Map

- Create: `/home/yanus/unified_ai_service/tests/__init__.py`
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_models.py`
- Create: `/home/yanus/unified_ai_service/tests/test_media_capabilities.py`
- Create: `/home/yanus/unified_ai_service/tests/test_voice_providers.py`
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_router.py`
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_executor.py`
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_endpoint.py`
- Create: `/home/yanus/unified_ai_service/multimodal_models.py`
- Create: `/home/yanus/unified_ai_service/media_capabilities.py`
- Create: `/home/yanus/unified_ai_service/voice_providers.py`
- Create: `/home/yanus/unified_ai_service/multimodal_router.py`
- Create: `/home/yanus/unified_ai_service/multimodal_executor.py`
- Modify: `/home/yanus/unified_ai_service/requirements.txt`
- Modify: `/home/yanus/unified_ai_service/media_audio.py`
- Modify: `/home/yanus/unified_ai_service/main.py`
- Modify: `/home/yanus/unified_ai_service/media_engine/README.md`

---

## Task 1: Data Models and Plan Validation

**Files:**
- Create: `/home/yanus/unified_ai_service/tests/__init__.py`
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_models.py`
- Create: `/home/yanus/unified_ai_service/multimodal_models.py`

- [ ] **Step 1: Create tests package**

Run:

```bash
cd /home/yanus/unified_ai_service
mkdir -p tests
touch tests/__init__.py
```

Expected: `tests/__init__.py` exists.

- [ ] **Step 2: Write failing model tests**

Create `/home/yanus/unified_ai_service/tests/test_multimodal_models.py`:

```python
import pytest

from multimodal_models import MediaAsset, MediaPlan, PlanValidationError


SUPPORTED = {"voice.tts", "image.generate", "package.bundle"}


def test_media_plan_validates_minimal_plan():
    plan = MediaPlan.from_dict(
        {
            "version": "1",
            "goal": "강의 음성 생성",
            "quality": "high",
            "steps": [
                {
                    "id": "step_1",
                    "action": "voice.tts",
                    "inputs": {"text": "안녕하세요"},
                    "outputs": {"audio": "lecture_voice"},
                }
            ],
            "final": {"primary": "lecture_voice", "format": "audio"},
        },
        supported_actions=SUPPORTED,
        asset_aliases=set(),
    )

    assert plan.goal == "강의 음성 생성"
    assert plan.steps[0].action == "voice.tts"
    assert plan.final["primary"] == "lecture_voice"


def test_media_plan_rejects_unknown_action():
    with pytest.raises(PlanValidationError, match="unknown action"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {
                        "id": "step_1",
                        "action": "video.teleport",
                        "inputs": {},
                        "outputs": {"video": "out"},
                    }
                ],
                "final": {"primary": "out", "format": "video"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_rejects_duplicate_step_ids():
    with pytest.raises(PlanValidationError, match="duplicate step id"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {"id": "same", "action": "voice.tts", "inputs": {"text": "a"}, "outputs": {"audio": "a"}},
                    {"id": "same", "action": "voice.tts", "inputs": {"text": "b"}, "outputs": {"audio": "b"}},
                ],
                "final": {"primary": "b", "format": "audio"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_rejects_duplicate_output_aliases():
    with pytest.raises(PlanValidationError, match="duplicate output alias"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {"id": "step_1", "action": "voice.tts", "inputs": {"text": "a"}, "outputs": {"audio": "same"}},
                    {"id": "step_2", "action": "image.generate", "inputs": {"prompt": "cat"}, "outputs": {"image": "same"}},
                ],
                "final": {"primary": "same", "format": "image"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_accepts_uploaded_asset_alias_reference():
    plan = MediaPlan.from_dict(
        {
            "version": "1",
            "goal": "image",
            "steps": [
                {
                    "id": "step_1",
                    "action": "image.generate",
                    "inputs": {"prompt": "based on asset:image_1"},
                    "outputs": {"image": "out_image"},
                }
            ],
            "final": {"primary": "out_image", "format": "image"},
        },
        supported_actions=SUPPORTED,
        asset_aliases={"image_1"},
    )

    assert plan.steps[0].inputs["prompt"] == "based on asset:image_1"


def test_media_asset_public_url_for_results_file():
    asset = MediaAsset(alias="voice", path="/home/yanus/unified_ai_service/results/a.wav", mime_type="audio/wav")

    assert asset.public_url() == "/api/results/a.wav"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'multimodal_models'`.

- [ ] **Step 4: Implement `multimodal_models.py`**

Create `/home/yanus/unified_ai_service/multimodal_models.py`:

```python
"""Data models for multimodal planning and execution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


RESULTS_DIR = "/home/yanus/unified_ai_service/results"


class PlanValidationError(ValueError):
    """Raised when a MediaPlan cannot be safely executed."""


@dataclass(frozen=True)
class MediaAsset:
    alias: str
    path: str
    mime_type: str = "application/octet-stream"
    filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "path": self.path,
            "mime_type": self.mime_type,
            "filename": self.filename or os.path.basename(self.path),
            "url": self.public_url(),
        }

    def public_url(self) -> str | None:
        abs_path = os.path.abspath(self.path)
        abs_results = os.path.abspath(RESULTS_DIR)
        if abs_path.startswith(abs_results + os.sep):
            return f"/api/results/{os.path.basename(abs_path)}"
        return None


@dataclass(frozen=True)
class MediaStep:
    id: str
    action: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    optional: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MediaStep":
        if not isinstance(raw, dict):
            raise PlanValidationError("step must be an object")
        step_id = raw.get("id")
        action = raw.get("action")
        if not isinstance(step_id, str) or not step_id.strip():
            raise PlanValidationError("step id is required")
        if not isinstance(action, str) or not action.strip():
            raise PlanValidationError("step action is required")
        inputs = raw.get("inputs", {})
        outputs = raw.get("outputs", {})
        if not isinstance(inputs, dict):
            raise PlanValidationError(f"step {step_id} inputs must be an object")
        if not isinstance(outputs, dict):
            raise PlanValidationError(f"step {step_id} outputs must be an object")
        for key, value in outputs.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
                raise PlanValidationError(f"step {step_id} outputs must map strings to aliases")
        return cls(
            id=step_id.strip(),
            action=action.strip(),
            inputs=inputs,
            outputs=outputs,
            optional=bool(raw.get("optional", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class MediaPlan:
    version: str
    goal: str
    steps: list[MediaStep]
    final: dict[str, Any]
    quality: str = "standard"
    channels: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        supported_actions: set[str],
        asset_aliases: set[str],
    ) -> "MediaPlan":
        if not isinstance(raw, dict):
            raise PlanValidationError("plan must be an object")
        if raw.get("version") != "1":
            raise PlanValidationError("version must be '1'")
        goal = raw.get("goal", "")
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError("goal is required")
        quality = raw.get("quality", "standard")
        if quality not in {"draft", "standard", "high"}:
            raise PlanValidationError("quality must be draft, standard, or high")
        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise PlanValidationError("steps must be a non-empty list")

        steps = [MediaStep.from_dict(step) for step in steps_raw]
        seen_ids: set[str] = set()
        aliases: set[str] = set(asset_aliases)
        produced_aliases: set[str] = set()
        for step in steps:
            if step.id in seen_ids:
                raise PlanValidationError(f"duplicate step id: {step.id}")
            seen_ids.add(step.id)
            if step.action not in supported_actions:
                raise PlanValidationError(f"unknown action: {step.action}")
            for alias in step.outputs.values():
                if alias in produced_aliases:
                    raise PlanValidationError(f"duplicate output alias: {alias}")
                produced_aliases.add(alias)
                aliases.add(alias)

        final = raw.get("final", {})
        if not isinstance(final, dict):
            raise PlanValidationError("final must be an object")
        primary = final.get("primary")
        if primary and primary not in aliases:
            raise PlanValidationError(f"final primary references missing alias: {primary}")
        channels = raw.get("channels", {})
        if not isinstance(channels, dict):
            raise PlanValidationError("channels must be an object")

        return cls(
            version="1",
            goal=goal.strip(),
            quality=quality,
            steps=steps,
            final=final,
            channels=channels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "goal": self.goal,
            "quality": self.quality,
            "steps": [step.to_dict() for step in self.steps],
            "final": self.final,
            "channels": self.channels,
        }
```

- [ ] **Step 5: Run model tests and verify they pass**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_models.py -v
```

Expected: all tests in `test_multimodal_models.py` PASS.

- [ ] **Step 6: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/tests/__init__.py unified_ai_service/tests/test_multimodal_models.py unified_ai_service/multimodal_models.py
git commit -m "feat(multimodal): add media plan data models"
```

---

## Task 2: Capability Registry

**Files:**
- Create: `/home/yanus/unified_ai_service/tests/test_media_capabilities.py`
- Create: `/home/yanus/unified_ai_service/media_capabilities.py`

- [ ] **Step 1: Write failing capability tests**

Create `/home/yanus/unified_ai_service/tests/test_media_capabilities.py`:

```python
from media_capabilities import SUPPORTED_ACTIONS, get_action, planner_prompt


def test_supported_actions_include_initial_multimodal_surface():
    assert "image.generate" in SUPPORTED_ACTIONS
    assert "image.analyze" in SUPPORTED_ACTIONS
    assert "voice.tts" in SUPPORTED_ACTIONS
    assert "ppt.generate" in SUPPORTED_ACTIONS
    assert "video.analyze" in SUPPORTED_ACTIONS
    assert "package.bundle" in SUPPORTED_ACTIONS


def test_get_action_returns_io_contract():
    action = get_action("voice.tts")

    assert action["kind"] == "voice"
    assert "text" in action["inputs"]
    assert action["outputs"] == {"audio": "audio file url"}


def test_planner_prompt_is_compact_and_mentions_schema_rules():
    prompt = planner_prompt()

    assert "Return JSON only" in prompt
    assert "voice.tts" in prompt
    assert "video.lipsync" in prompt
    assert "Unsupported actions are invalid" in prompt
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_media_capabilities.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'media_capabilities'`.

- [ ] **Step 3: Implement `media_capabilities.py`**

Create `/home/yanus/unified_ai_service/media_capabilities.py`:

```python
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
    "ppt.generate": {
        "kind": "document",
        "description": "Create a PowerPoint deck from a topic or outline.",
        "inputs": {"topic": "string"},
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
        "inputs": {"control_image": "asset alias", "prompt": "string", "control_type": "optional string", "strength": "optional float"},
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
    "voice.tts": {
        "kind": "voice",
        "description": "Generate spoken narration from text using local F5-TTS or ElevenLabs when configured.",
        "inputs": {"text": "string", "provider": "optional auto|local_f5|elevenlabs", "voice": "optional string"},
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
        "description": "Lip-sync a presenter video to narration. Registered now; full engine lands in Phase B4.",
        "inputs": {"video": "asset alias", "audio": "asset alias"},
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
        "Schema: {version:'1', goal:string, quality:'draft|standard|high', steps:[{id, action, inputs, outputs}], final:{primary, format}}",
        "Available actions:",
    ]
    for name in sorted(CAPABILITIES):
        cap = CAPABILITIES[name]
        lines.append(f"- {name}: {cap['description']} Inputs={cap['inputs']} Outputs={cap['outputs']}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run capability tests and verify they pass**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_media_capabilities.py -v
```

Expected: all tests in `test_media_capabilities.py` PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/tests/test_media_capabilities.py unified_ai_service/media_capabilities.py
git commit -m "feat(multimodal): add capability registry"
```

---

## Task 3: Voice Provider Abstraction with ElevenLabs Optional Fallback

**Files:**
- Modify: `/home/yanus/unified_ai_service/requirements.txt`
- Create: `/home/yanus/unified_ai_service/tests/test_voice_providers.py`
- Create: `/home/yanus/unified_ai_service/voice_providers.py`
- Modify: `/home/yanus/unified_ai_service/media_audio.py`

- [ ] **Step 1: Add httpx dependency**

Modify `/home/yanus/unified_ai_service/requirements.txt` and add this line if absent:

```text
httpx>=0.27
```

- [ ] **Step 2: Install dependency**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/pip install "httpx>=0.27"
```

Expected: pip exits with code 0.

- [ ] **Step 3: Write failing voice provider tests**

Create `/home/yanus/unified_ai_service/tests/test_voice_providers.py`:

```python
from pathlib import Path

import pytest

import voice_providers


def test_choose_provider_auto_without_key_uses_local(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    assert voice_providers.choose_provider("auto", quality="high") == "local_f5"


def test_choose_provider_auto_with_key_and_high_quality_uses_elevenlabs(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")

    assert voice_providers.choose_provider("auto", quality="high") == "elevenlabs"


def test_split_text_preserves_all_content():
    text = "첫 문장입니다. 둘째 문장입니다.\n\n새 문단입니다."

    chunks = voice_providers.split_text_for_tts(text, limit=12)

    assert "".join(chunk.text for chunk in chunks).replace("\n\n", "") == text.replace("\n\n", "")
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_synthesize_auto_falls_back_to_local(monkeypatch, tmp_path):
    async def fake_local(text, ref_audio, ref_text, output_path):
        Path(output_path).write_bytes(b"local wav")
        return output_path

    async def fake_eleven(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setattr(voice_providers, "_synthesize_local_f5", fake_local)
    monkeypatch.setattr(voice_providers, "_synthesize_elevenlabs", fake_eleven)

    result = await voice_providers.synthesize_speech(
        "긴 강의 대본",
        provider="auto",
        quality="high",
        output_path=str(tmp_path / "out.wav"),
    )

    assert result["provider"] == "local_f5"
    assert result["url"].endswith("out.wav") or result["path"].endswith("out.wav")


@pytest.mark.asyncio
async def test_synthesize_forced_elevenlabs_failure_raises(monkeypatch, tmp_path):
    async def fake_eleven(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setattr(voice_providers, "_synthesize_elevenlabs", fake_eleven)

    with pytest.raises(RuntimeError, match="network down"):
        await voice_providers.synthesize_speech(
            "text",
            provider="elevenlabs",
            output_path=str(tmp_path / "out.mp3"),
        )
```

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_voice_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'voice_providers'`.

- [ ] **Step 5: Implement `voice_providers.py`**

Create `/home/yanus/unified_ai_service/voice_providers.py`:

```python
"""TTS provider abstraction with optional ElevenLabs fallback."""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydub import AudioSegment


BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass(frozen=True)
class TTSChunk:
    index: int
    text: str
    context_before: str = ""
    context_after: str = ""


def _public_url(path: str) -> str:
    return f"/api/results/{os.path.basename(path)}"


def choose_provider(requested: str = "auto", quality: str = "standard") -> str:
    requested = requested or "auto"
    if requested not in {"auto", "local_f5", "elevenlabs"}:
        raise ValueError("provider must be auto, local_f5, or elevenlabs")
    if requested != "auto":
        return requested
    has_elevenlabs = bool(os.getenv("ELEVENLABS_API_KEY") and os.getenv("ELEVENLABS_VOICE_ID"))
    if quality == "high" and has_elevenlabs:
        return "elevenlabs"
    return "local_f5"


def split_text_for_tts(text: str, limit: int = 2500) -> list[TTSChunk]:
    normalized = text.strip()
    if not normalized:
        return [TTSChunk(index=0, text="")]
    parts = re.split(r"(?<=[.!?。！？다요죠음])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        candidate = f"{current} {part}".strip() if current else part
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
    voice_id = voice if voice and voice != "default" else os.getenv("ELEVENLABS_VOICE_ID")
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
```

- [ ] **Step 6: Route `media_audio` TTS through provider**

Modify `/home/yanus/unified_ai_service/media_audio.py` by adding this function near the TTS section, leaving `generate_tts_with_effects` intact:

```python
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

    return await voice_providers.synthesize_speech(
        text,
        provider=provider,
        quality=quality,
        voice=voice,
        ref_audio=ref_audio,
        ref_text=ref_text,
        output_path=output_path,
    )
```

- [ ] **Step 7: Run voice provider tests**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_voice_providers.py -v
```

Expected: all tests in `test_voice_providers.py` PASS.

- [ ] **Step 8: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/requirements.txt unified_ai_service/tests/test_voice_providers.py unified_ai_service/voice_providers.py unified_ai_service/media_audio.py
git commit -m "feat(multimodal): add optional ElevenLabs voice provider"
```

---

## Task 4: Router with LLM JSON Planning and Rule Fallback

**Files:**
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_router.py`
- Create: `/home/yanus/unified_ai_service/multimodal_router.py`

- [ ] **Step 1: Write failing router tests**

Create `/home/yanus/unified_ai_service/tests/test_multimodal_router.py`:

```python
import pytest

from multimodal_models import MediaAsset
import multimodal_router


@pytest.mark.asyncio
async def test_router_parses_clean_llm_json(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return """
        {"version":"1","goal":"tts","quality":"high","steps":[{"id":"step_1","action":"voice.tts","inputs":{"text":"hello"},"outputs":{"audio":"voice"}}],"final":{"primary":"voice","format":"audio"}}
        """

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("읽어줘", assets=[], quality="high")

    assert plan.steps[0].action == "voice.tts"
    assert plan.quality == "high"


@pytest.mark.asyncio
async def test_router_strips_markdown_fence(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return """```json
        {"version":"1","goal":"image","steps":[{"id":"step_1","action":"image.generate","inputs":{"prompt":"cat"},"outputs":{"image":"img"}}],"final":{"primary":"img","format":"image"}}
        ```"""

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("고양이 이미지", assets=[])

    assert plan.steps[0].action == "image.generate"


@pytest.mark.asyncio
async def test_router_falls_back_on_invalid_json(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return "not json"

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("강의 음성으로 읽어줘", assets=[])

    assert plan.steps[0].action == "voice.tts"


def test_rule_fallback_maps_image_analysis():
    asset = MediaAsset(alias="image_1", path="/tmp/a.png", mime_type="image/png")

    plan = multimodal_router.fallback_plan("이 이미지를 분석해줘", [asset], quality="standard")

    assert plan.steps[0].action == "image.analyze"
    assert plan.steps[0].inputs["image"] == "image_1"


def test_rule_fallback_maps_ppt_generation():
    plan = multimodal_router.fallback_plan("AI 윤리 발표자료 만들어줘", [], quality="standard")

    assert plan.steps[0].action == "ppt.generate"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_router.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'multimodal_router'`.

- [ ] **Step 3: Implement `multimodal_router.py`**

Create `/home/yanus/unified_ai_service/multimodal_router.py`:

```python
"""Natural-language multimodal planner."""
from __future__ import annotations

import json
import re
from typing import Any

import llm_service
from media_capabilities import SUPPORTED_ACTIONS, planner_prompt
from multimodal_models import MediaAsset, MediaPlan, PlanValidationError


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
) -> MediaPlan:
    try:
        raw = await _llm_plan(instruction, assets, quality, preferred_voice_provider)
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
```

- [ ] **Step 4: Run router tests**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_router.py -v
```

Expected: all tests in `test_multimodal_router.py` PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/tests/test_multimodal_router.py unified_ai_service/multimodal_router.py
git commit -m "feat(multimodal): add natural language planner"
```

---

## Task 5: Sequential Executor and Real Action Dispatch

**Files:**
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_executor.py`
- Create: `/home/yanus/unified_ai_service/multimodal_executor.py`

- [ ] **Step 1: Write failing executor tests**

Create `/home/yanus/unified_ai_service/tests/test_multimodal_executor.py`:

```python
from pathlib import Path

import pytest

from multimodal_models import MediaAsset, MediaPlan
import multimodal_executor


def _plan(raw):
    return MediaPlan.from_dict(
        raw,
        supported_actions=multimodal_executor.SUPPORTED_ACTIONS,
        asset_aliases={"image_1", "audio_1"},
    )


@pytest.mark.asyncio
async def test_executor_chains_text_output(monkeypatch):
    async def fake_text(inputs, ctx):
        return {"text": "generated text"}

    async def fake_tts(inputs, ctx):
        assert inputs["text"] == "generated text"
        return {"audio": "/api/results/voice.wav"}

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "text.generate", fake_text)
    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "voice.tts", fake_tts)
    plan = _plan(
        {
            "version": "1",
            "goal": "chain",
            "steps": [
                {"id": "s1", "action": "text.generate", "inputs": {"prompt": "write"}, "outputs": {"text": "script"}},
                {"id": "s2", "action": "voice.tts", "inputs": {"text": "$script"}, "outputs": {"audio": "voice"}},
            ],
            "final": {"primary": "voice", "format": "audio"},
        }
    )

    result = await multimodal_executor.execute_plan(plan, assets=[])

    assert result["final"]["url"] == "/api/results/voice.wav"
    assert result["steps"][1]["status"] == "completed"


@pytest.mark.asyncio
async def test_executor_records_step_failure(monkeypatch):
    async def fail(inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "text.generate", fail)
    plan = _plan(
        {
            "version": "1",
            "goal": "bad",
            "steps": [{"id": "s1", "action": "text.generate", "inputs": {"prompt": "x"}, "outputs": {"text": "out"}}],
            "final": {"primary": "out", "format": "text"},
        }
    )

    with pytest.raises(multimodal_executor.StepExecutionError, match="s1"):
        await multimodal_executor.execute_plan(plan, assets=[])


@pytest.mark.asyncio
async def test_path_result_is_converted_to_url(tmp_path):
    out = tmp_path / "x.png"
    out.write_bytes(b"png")

    assert multimodal_executor._normalize_result({"image": str(out)})["image"] == str(out)

    result_path = Path("/home/yanus/unified_ai_service/results/x.png")
    normalized = multimodal_executor._normalize_result({"image": result_path})

    assert normalized["image"] == "/api/results/x.png"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_executor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'multimodal_executor'`.

- [ ] **Step 3: Implement `multimodal_executor.py`**

Create `/home/yanus/unified_ai_service/multimodal_executor.py`:

```python
"""Sequential executor for validated MediaPlan objects."""
from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import llm_service
import media_audio
import media_image
import media_video
import ppt_service
import stt_service
from media_capabilities import SUPPORTED_ACTIONS
from multimodal_models import MediaAsset, MediaPlan, MediaStep


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
    text = await llm_service.generate_text(
        inputs.get("prompt", ""),
        inputs.get("system_prompt", "You are a helpful multimodal media assistant."),
    )
    return {"text": text}


async def _handle_ppt_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    topic = inputs.get("topic", "")
    slides = await llm_service.generate_ppt_structure(topic)
    output_path = os.path.join(RESULTS_DIR, f"presentation_{uuid.uuid4().hex[:8]}.pptx")
    ppt_service.generate_ppt_file(topic, slides, output_path)
    return {"ppt": output_path}


async def _handle_image_generate(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = await media_image.generate_image(
        inputs.get("prompt", ""),
        workflow=inputs.get("workflow", "zimage_turbo"),
    )
    return {"image": path}


async def _handle_image_edit(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = await media_image.edit_image(inputs["image"], inputs.get("prompt", ""))
    return {"image": path}


async def _handle_image_control(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = await media_image.control_image(
        inputs.get("prompt", ""),
        inputs["control_image"],
        control_type=inputs.get("control_type", "canny"),
        strength=float(inputs.get("strength", 0.7)),
    )
    return {"image": path}


async def _handle_image_inpaint(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = await media_image.inpaint_image(inputs["image"], inputs["mask"], inputs.get("prompt", ""))
    return {"image": path}


async def _handle_image_analyze(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    data_url = _file_to_data_url(inputs["image"], fallback_mime="image/png")
    text = await llm_service.analyze_image(data_url, inputs.get("prompt", "Analyze this image."))
    return {"text": text}


async def _handle_audio_music(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    duration = int(inputs.get("duration", 10))
    if duration > 30:
        path = await media_audio.generate_long_music(inputs.get("prompt", ""), duration)
    else:
        path = await media_audio.generate_music(inputs.get("prompt", ""), duration)
    return {"audio": path}


async def _handle_audio_transcribe(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    return {"text": stt_service.transcribe_audio(inputs["audio"])}


async def _handle_voice_tts(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
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
    path = await media_video.generate_long_video(
        inputs.get("prompt", ""),
        inputs["image"],
        total_duration_target=int(inputs.get("duration", 30)),
    )
    return {"video": path}


async def _handle_video_edit(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    path = await media_video.edit_video(
        inputs["video"],
        audio_path=inputs.get("audio"),
        image_path=inputs.get("image"),
        prompt=inputs.get("prompt", ""),
    )
    return {"video": path}


async def _handle_video_analyze(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    return {"text": await media_video.analyze_video(inputs["video"], inputs.get("prompt", ""))}


async def _handle_video_shorts(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    return {"video": await media_video.shorten_video(inputs["video"], inputs.get("prompt", ""))}


async def _handle_video_lipsync(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    raise RuntimeError("video.lipsync is registered but requires Phase B4 lip-sync engine installation")


async def _handle_package_bundle(inputs: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
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
```

- [ ] **Step 4: Run executor tests**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_executor.py -v
```

Expected: all tests in `test_multimodal_executor.py` PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/tests/test_multimodal_executor.py unified_ai_service/multimodal_executor.py
git commit -m "feat(multimodal): add sequential plan executor"
```

---

## Task 6: FastAPI Endpoint and Job Wiring

**Files:**
- Create: `/home/yanus/unified_ai_service/tests/test_multimodal_endpoint.py`
- Modify: `/home/yanus/unified_ai_service/main.py`

- [ ] **Step 1: Write failing endpoint tests**

Create `/home/yanus/unified_ai_service/tests/test_multimodal_endpoint.py`:

```python
import pytest

import main


@pytest.mark.asyncio
async def test_save_multimodal_uploads_assigns_aliases(tmp_path, monkeypatch):
    class FakeUpload:
        filename = "a.png"
        content_type = "image/png"

        async def read(self):
            return b"png"

    monkeypatch.setattr(main, "UPLOADS_DIR", str(tmp_path))

    assets = await main._save_multimodal_uploads([FakeUpload()])

    assert assets[0].alias == "image_1"
    assert assets[0].path.endswith("a.png")


@pytest.mark.asyncio
async def test_process_multimodal_task_updates_job(monkeypatch):
    updates = []

    async def fake_plan_request(instruction, assets, quality, preferred_voice_provider):
        from multimodal_models import MediaPlan
        return MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "text",
                "steps": [{"id": "s1", "action": "text.generate", "inputs": {"prompt": "x"}, "outputs": {"text": "answer"}}],
                "final": {"primary": "answer", "format": "text"},
            },
            supported_actions={"text.generate"},
            asset_aliases=set(),
        )

    async def fake_execute_plan(plan, assets):
        return {"plan": plan.to_dict(), "steps": [], "final": {"type": "text", "value": "ok"}}

    monkeypatch.setattr(main.multimodal_router, "plan_request", fake_plan_request)
    monkeypatch.setattr(main.multimodal_executor, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(main.job_manager, "update_job", lambda *args, **kwargs: updates.append((args, kwargs)))

    await main.process_multimodal_task("job1", "hello", [], "standard", "auto")

    assert updates[0][0] == ("job1", "processing")
    assert updates[-1][0][0] == "job1"
    assert updates[-1][0][1] == "completed"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_endpoint.py -v
```

Expected: FAIL because `main` has no `_save_multimodal_uploads` or `process_multimodal_task`.

- [ ] **Step 3: Add imports to `main.py`**

Modify `/home/yanus/unified_ai_service/main.py` near existing media imports:

```python
import multimodal_router
import multimodal_executor
from multimodal_models import MediaAsset
```

- [ ] **Step 4: Add helper functions to `main.py`**

Add this block before the endpoint section in `/home/yanus/unified_ai_service/main.py`:

```python
def _asset_alias_for_upload(index: int, upload: UploadFile) -> str:
    mime = upload.content_type or "application/octet-stream"
    if mime.startswith("image/"):
        prefix = "image"
    elif mime.startswith("audio/"):
        prefix = "audio"
    elif mime.startswith("video/"):
        prefix = "video"
    else:
        prefix = "file"
    return f"{prefix}_{index}"


async def _save_multimodal_uploads(files: Optional[List[UploadFile]]) -> List[MediaAsset]:
    assets: List[MediaAsset] = []
    if not files:
        return assets
    counters = {"image": 0, "audio": 0, "video": 0, "file": 0}
    for upload in files:
        mime = upload.content_type or "application/octet-stream"
        if mime.startswith("image/"):
            kind = "image"
        elif mime.startswith("audio/"):
            kind = "audio"
        elif mime.startswith("video/"):
            kind = "video"
        else:
            kind = "file"
        counters[kind] += 1
        alias = f"{kind}_{counters[kind]}"
        safe_name = os.path.basename(upload.filename or f"{alias}.bin")
        path = os.path.join(UPLOADS_DIR, f"multi_{uuid.uuid4().hex}_{safe_name}")
        with open(path, "wb") as f:
            f.write(await upload.read())
        assets.append(MediaAsset(alias=alias, path=path, mime_type=mime, filename=safe_name))
    return assets


async def process_multimodal_task(
    job_id: str,
    instruction: str,
    assets: List[MediaAsset],
    quality: str,
    preferred_voice_provider: str,
):
    try:
        job_manager.update_job(job_id, "processing")
        plan = await multimodal_router.plan_request(
            instruction,
            assets,
            quality=quality,
            preferred_voice_provider=preferred_voice_provider,
        )
        result = await multimodal_executor.execute_plan(plan, assets)
        job_manager.update_job(job_id, "completed", result=result)
    except Exception as e:
        logger.error("Multimodal job failed: %s", e, exc_info=True)
        job_manager.update_job(job_id, "failed", error=str(e))
```

- [ ] **Step 5: Add `/api/multimodal/execute` endpoint**

Add this endpoint in `/home/yanus/unified_ai_service/main.py` near other media endpoints:

```python
@app.post("/api/multimodal/execute")
async def multimodal_execute_endpoint(
    background_tasks: BackgroundTasks,
    instruction: str = Form(...),
    quality: str = Form("standard"),
    preferred_voice_provider: str = Form("auto"),
    files: Optional[List[UploadFile]] = File(None),
    auth = Depends(flexible_auth),
):
    if quality not in {"draft", "standard", "high"}:
        raise HTTPException(status_code=400, detail="quality must be draft, standard, or high")
    if preferred_voice_provider not in {"auto", "local_f5", "elevenlabs"}:
        raise HTTPException(status_code=400, detail="preferred_voice_provider must be auto, local_f5, or elevenlabs")
    assets = await _save_multimodal_uploads(files)
    job_id = job_manager.create_job(
        "multimodal",
        {
            "instruction": instruction,
            "quality": quality,
            "preferred_voice_provider": preferred_voice_provider,
            "assets": [asset.to_dict() for asset in assets],
        },
    )
    background_tasks.add_task(process_multimodal_task, job_id, instruction, assets, quality, preferred_voice_provider)
    return {"job_id": job_id}
```

- [ ] **Step 6: Run endpoint tests**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/test_multimodal_endpoint.py -v
```

Expected: all tests in `test_multimodal_endpoint.py` PASS.

- [ ] **Step 7: Import check**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python - <<'PY'
import main
paths = sorted(route.path for route in main.app.routes)
assert "/api/multimodal/execute" in paths
print("multimodal endpoint registered")
PY
```

Expected: prints `multimodal endpoint registered`.

- [ ] **Step 8: Commit**

Run:

```bash
cd /home/yanus
git add unified_ai_service/tests/test_multimodal_endpoint.py unified_ai_service/main.py
git commit -m "feat(multimodal): expose multimodal execute endpoint"
```

---

## Task 7: Documentation and Verification

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_engine/README.md`

- [ ] **Step 1: Update README**

Modify `/home/yanus/unified_ai_service/media_engine/README.md` by adding this section before `## 알려진 한계`:

```markdown
## Phase B Core 산출 (2026-05-23)

- 신규 endpoint:
  - `POST /api/multimodal/execute`
- 입력:
  - `instruction`: 자연어 작업 지시
  - `quality`: `draft` / `standard` / `high`
  - `preferred_voice_provider`: `auto` / `local_f5` / `elevenlabs`
  - `files`: 이미지, 오디오, 비디오 첨부 파일 목록
- 내부 동작:
  - LLM planner가 `MediaPlan` JSON 생성
  - invalid planner output 또는 vLLM unavailable 시 rule-based fallback
  - `multimodal_executor`가 step을 순차 실행
  - job result에 실행 plan, step 결과, final 결과 저장
- ElevenLabs 설정:
  - `ELEVENLABS_API_KEY`: ElevenLabs API key
  - `ELEVENLABS_VOICE_ID`: 기본 voice id
  - `ELEVENLABS_MODEL_ID`: 기본 `eleven_multilingual_v2`
  - `ELEVENLABS_OUTPUT_FORMAT`: 기본 `mp3_44100_128`
- 로컬 fallback:
  - key가 없으면 `auto` provider는 `local_f5`로 실행
  - ElevenLabs 장애 시 `auto`는 로컬 fallback
  - `preferred_voice_provider=elevenlabs`로 강제했을 때만 ElevenLabs 실패가 job 실패가 됨
```

- [ ] **Step 2: Run full unit tests**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest tests/ media_engine/tests/ -v
```

Expected: `tests/` pass and existing `media_engine/tests/` still pass. Integration tests without `--integration` are skipped.

- [ ] **Step 3: Run endpoint smoke without starting a server**

Run:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python - <<'PY'
import asyncio
from multimodal_router import fallback_plan
from multimodal_models import MediaAsset

async def main():
    p1 = fallback_plan("고양이 이미지를 만들어줘", [], quality="standard")
    assert p1.steps[0].action == "image.generate"
    p2 = fallback_plan("이 이미지를 분석해줘", [MediaAsset(alias="image_1", path="/tmp/a.png", mime_type="image/png")])
    assert p2.steps[0].action == "image.analyze"
    print("router smoke ok")

asyncio.run(main())
PY
```

Expected: prints `router smoke ok`.

- [ ] **Step 4: Check git diff**

Run:

```bash
cd /home/yanus
git diff --stat HEAD
```

Expected: only README changes after previous task commits.

- [ ] **Step 5: Commit README**

Run:

```bash
cd /home/yanus
git add unified_ai_service/media_engine/README.md
git commit -m "docs(multimodal): document multimodal execute endpoint"
```

---

## Task 8: Optional Manual Smoke on Running Service

**Files:**
- No source changes.

- [ ] **Step 1: Restart service if this repo is deployed as the active service**

Run the environment-specific restart command already used for this machine. If no service manager is configured, skip this step and run the app manually in a shell:

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python main.py
```

Expected: FastAPI process listens on port `8081`.

- [ ] **Step 2: Submit text-only multimodal request**

Run:

```bash
curl -s -X POST http://localhost:8081/api/multimodal/execute \
  -F 'instruction=AI 윤리 발표자료 개요를 만들어줘' \
  -F 'quality=draft'
```

Expected: JSON with `job_id`.

- [ ] **Step 3: Poll job**

Replace `<job_id>` and run:

```bash
curl -s http://localhost:8081/api/jobs/<job_id>
```

Expected: status becomes `completed` or `failed` with a clear step error. A failed LLM/vLLM response is acceptable in this manual smoke only if the job error explains the unavailable dependency.

- [ ] **Step 4: Submit local fallback TTS request**

Run:

```bash
curl -s -X POST http://localhost:8081/api/multimodal/execute \
  -F 'instruction=안녕하세요. 이 문장을 강의 음성으로 읽어줘' \
  -F 'quality=standard' \
  -F 'preferred_voice_provider=auto'
```

Expected: JSON with `job_id`; job uses `local_f5` when ElevenLabs env vars are absent.

---

## Self-Review

- **Spec coverage:** This plan implements data models, capability registry, planner, fallback, executor, ElevenLabs optional provider, endpoint wiring, docs, and verification from the Phase B Core spec.
- **Deferred but represented:** `video.lipsync`, WhisperX/pyannote, YOLO tracking, and SAM 2 are represented in schema/capabilities but not installed here. The spec marks them as follow-up phases.
- **Placeholder scan:** The plan contains no TBD/TODO steps. Unsupported future engines fail explicitly with actionable messages.
- **Type consistency:** `MediaAsset`, `MediaPlan`, `MediaStep`, `StepExecutionError`, `plan_request`, and `execute_plan` names are consistent across tests and implementation snippets.
- **Test strategy:** Every production module is introduced by a failing test first. Full verification includes new tests plus existing `media_engine` tests.
