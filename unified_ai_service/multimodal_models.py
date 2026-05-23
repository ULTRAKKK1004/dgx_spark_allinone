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
