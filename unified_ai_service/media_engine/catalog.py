"""워크플로우 메타데이터 단일 출처.

각 워크플로우는 다음 키를 가진다:
- template: workflows/ 디렉토리 내 .json.j2 파일명
- params: {파라미터명: (타입, 기본값)} — 기본값이 `...` (Ellipsis)이면 필수
- models_required: 디스크상에 존재해야 하는 모델 파일 (MODELS_ROOT 기준 상대경로)
- output_node: ComfyUI history outputs 에서 결과를 꺼낼 노드 id
- vram_class: "light" | "heavy" ("heavy"는 vLLM swap 필요)
- timeout_sec: ComfyUI 호출 최대 대기 시간
"""

import os
from pathlib import Path
from typing import Any

MODELS_ROOT = "/home/yanus/Docker/models"

WORKFLOWS: dict[str, dict[str, Any]] = {
    "image.gen.zimage_turbo": {
        "template": "image_gen_zimage_turbo.json.j2",
        "params": {
            "prompt":  (str,   ...),
            "width":   (int,   1024),
            "height":  (int,   1024),
            "steps":   (int,   8),
            "seed":    (int,   0),
        },
        "models_required": [
            "diffusion_models/z_image_turbo_bf16.safetensors",
            "text_encoders/qwen_3_4b.safetensors",
            "vae/ae.safetensors",
        ],
        "output_node": "9",
        "vram_class": "light",
        "timeout_sec": 90,
    },
    "image.edit.qwen": {
        "template": "image_edit_qwen.json.j2",
        "params": {
            "prompt":     (str, ...),
            "image_name": (str, ...),
            "steps":      (int, 20),
            "seed":       (int, 0),
            "denoise":    (float, 0.85),
        },
        "models_required": [
            "diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "vae/qwen_image_vae.safetensors",
        ],
        "output_node": "9",
        "vram_class": "heavy",
        "timeout_sec": 300,
    },
    "video.i2v.wan22": {
        "template": "video_i2v_wan22.json.j2",
        "params": {
            "prompt":     (str, ...),
            "image_name": (str, ...),
            "frames":     (int, 81),
            "steps":      (int, 4),
            "seed":       (int, 0),
            "width":      (int, 720),
            "height":     (int, 720),
        },
        "models_required": [
            "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae/wan_2.1_vae.safetensors",
        ],
        "output_node": "combine",
        "vram_class": "heavy",
        "timeout_sec": 600,
    },
    "video.s2v.wan22": {
        "template": "video_s2v_wan22.json.j2",
        "params": {
            "prompt":      (str, ...),
            "image_name":  (str, ...),
            "audio_name":  (str, ...),
            "frames":      (int, 81),
            "steps":       (int, 10),
            "seed":        (int, 0),
            "width":       (int, 640),
            "height":      (int, 640),
        },
        "models_required": [
            "diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors",
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "vae/wan_2.1_vae.safetensors",
            "audio_encoders/wav2vec2_large_english_fp16.safetensors",
        ],
        "output_node": "combine",
        "vram_class": "heavy",
        "timeout_sec": 900,
    },
}


def list_workflows() -> list[str]:
    return list(WORKFLOWS.keys())


def get(workflow_id: str) -> dict[str, Any]:
    if workflow_id not in WORKFLOWS:
        raise KeyError(f"unknown workflow: {workflow_id}")
    return WORKFLOWS[workflow_id]


def validate(meta: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """타입 강제·기본값 채움. 필수 누락/타입 변환 실패는 ValueError."""
    out: dict[str, Any] = {}
    spec = meta["params"]
    for name, (typ, default) in spec.items():
        if name in params:
            value = params[name]
            if value is None:
                raise ValueError(f"param {name!r}: cannot be None")
            if isinstance(value, bool) and typ is int:
                raise ValueError(f"param {name!r}: bool not accepted for int")
            try:
                out[name] = typ(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"param {name!r}: cannot coerce to {typ.__name__}: {e}") from e
        elif default is Ellipsis:
            raise ValueError(f"missing required param: {name!r}")
        else:
            out[name] = default
    extra = set(params) - set(spec)
    if extra:
        raise ValueError(f"unknown params: {sorted(extra)}")
    return out


def check_models_present(meta: dict[str, Any]) -> list[str]:
    """누락된 모델 파일의 상대경로 리스트 반환."""
    missing: list[str] = []
    root = Path(MODELS_ROOT).resolve()
    for rel in meta["models_required"]:
        if os.path.isabs(rel) or ".." in Path(rel).parts:
            raise ValueError(f"models_required entry must be relative without '..': {rel!r}")
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"models_required escapes MODELS_ROOT: {rel!r}")
        if not path.exists():
            missing.append(rel)
    return missing
