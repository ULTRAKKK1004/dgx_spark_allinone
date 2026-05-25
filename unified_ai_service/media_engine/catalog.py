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
    "image.gen.flux": {
        "template": "image_gen_flux.json.j2",
        "params": {
            "prompt":   (str,   ...),
            "width":    (int,   1024),
            "height":   (int,   1024),
            "steps":    (int,   20),
            "seed":     (int,   0),
            "guidance": (float, 3.5),
            "workflow": (str, "dev"),
        },
        "models_required": [
            "diffusion_models/FLUX1/flux1-dev-fp8.safetensors",
            "diffusion_models/FLUX1/flux1-schnell-fp8.safetensors",
            "text_encoders/clip_l.safetensors",
            "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
            "vae/ae.safetensors",
        ],
        "output_node": "9",
        "vram_class": "heavy",
        "timeout_sec": 600,
        "choices": {
            "workflow": {"dev", "schnell"},
        },
    },
    "image.ctrl.flux_union": {
        "template": "image_ctrl_flux_union.json.j2",
        "params": {
            "prompt":        (str,   ...),
            "control_image": (str,   ...),
            "control_type":  (str,   "canny"),
            "strength":      (float, 0.7),
            "width":         (int,   1024),
            "height":        (int,   1024),
            "steps":         (int,   20),
            "seed":          (int,   0),
            "guidance":      (float, 3.5),
        },
        "models_required": [
            "diffusion_models/FLUX1/flux1-dev-fp8.safetensors",
            "controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors",
            "text_encoders/clip_l.safetensors",
            "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
            "vae/ae.safetensors",
        ],
        "output_node": "9",
        "vram_class": "heavy",
        "timeout_sec": 600,
        "choices": {
            "control_type": {"canny", "openpose", "depth", "scribble"},
        },
    },
    "image.inpaint.qwen": {
        "template": "image_inpaint_qwen.json.j2",
        "params": {
            "prompt":     (str,   ...),
            "image_name": (str,   ...),
            "mask_name":  (str,   ...),
            "steps":      (int,   20),
            "seed":       (int,   0),
            "denoise":    (float, 0.9),
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
    "image.analyze.janus": {
        "template": "image_analyze_janus.json.j2",
        "params": {
            "image_name":     (str, ...),
            "prompt":         (str, "Describe this image in detail."),
            "seed":           (int, 666),
            "temperature":    (float, 0.1),
            "max_new_tokens": (int, 512),
        },
        "models_required": [
            "Janus-Pro/Janus-Pro-7B/config.json",
            "Janus-Pro/Janus-Pro-7B/pytorch_model-00001-of-00002.bin",
            "Janus-Pro/Janus-Pro-7B/pytorch_model-00002-of-00002.bin",
        ],
        "output_node": "analyze",
        "vram_class": "heavy",
        "timeout_sec": 300,
    },
    "video.lipsync.liveportrait": {
        "template": "video_lipsync_liveportrait.json.j2",
        "params": {
            "image_name": (str, ...),
            "audio_name": (str, ...),
            "fps":        (int, 25),
        },
        "models_required": [
            "liveportrait/appearance_feature_extractor.safetensors",
            "liveportrait/motion_extractor.safetensors",
            "liveportrait/warping_module.safetensors",
            "liveportrait/spade_generator.safetensors",
            "liveportrait/stitching_retargeting_module.safetensors",
        ],
        "output_node": "combine",
        "vram_class": "heavy",
        "timeout_sec": 300,
    },
    "video.lipsync.wav2lip": {
        "template": "video_lipsync_wav2lip.json.j2",
        "params": {
            "image_name": (str, ...),
            "audio_name": (str, ...),
            "face_detect_batch": (int, 8),
        },
        "models_required": [
            "wav2lip/wav2lip_gan.pth",
        ],
        "output_node": "combine",
        "vram_class": "heavy",
        "timeout_sec": 300,
    },
    "image.face_swap.reactor": {
        "template": "image_face_swap_reactor.json.j2",
        "params": {
            "source_image": (str, ...),
            "input_image":  (str, ...),
        },
        "models_required": [
            "insightface/inswapper_128.onnx",
        ],
        "output_node": "save_image",
        "vram_class": "heavy",
        "timeout_sec": 60,
    },
    "video.idle_loop.animatediff": {
        "template": "video_idle_loop_animatediff.json.j2",
        "params": {
            "image_name": (str, ...),
            "prompt":     (str, "subtle natural breathing, blinking"),
        },
        "models_required": [
            "animatediff_models/mm_sdxl_v10_beta.ckpt",
        ],
        "output_node": "combine",
        "vram_class": "heavy",
        "timeout_sec": 300,
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

    choices = meta.get("choices", {})
    for name, allowed in choices.items():
        if name in out and out[name] not in allowed:
            allowed_text = "|".join(sorted(str(v) for v in allowed))
            raise ValueError(f"param {name!r}: must be one of {allowed_text}")
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
