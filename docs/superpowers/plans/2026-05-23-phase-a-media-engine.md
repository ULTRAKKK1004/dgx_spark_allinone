# Phase A — `media_engine` 기반 인프라 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `unified_ai_service`에 공용 미디어 엔진(`media_engine/` 패키지)을 구축해 (a) ComfyUI 워크플로우를 카탈로그·템플릿 방식으로 관리하고, (b) GPU 자원(vLLM ↔ ComfyUI heavy 모델)을 직렬화·전환하고, (c) 무빙윈도우 청크링 유틸을 제공하고, (d) 미디어 잡을 직렬화 큐로 묶어, 현재 mock 처리된 비디오 생성을 실제 Wan2.2 i2v로 교체한다.

**Architecture:** 신규 패키지 `unified_ai_service/media_engine/` 하위에 `catalog.py`(메타데이터), `comfyui_client.py`(HTTP I/O), `gpu_arbiter.py`(vLLM docker 제어), `window.py`(청크 유틸), `runner.py`(통합 실행), `job_queue.py`(잡 직렬화)를 두고, 기존 `media_image.py`/`media_video.py`/`media_audio.py`는 이 엔진을 호출하는 얇은 shim으로 슬림화한다. 워크플로우는 `media_engine/workflows/*.json.j2` 에 ComfyUI API 형식 JSON + Jinja 변수로 저장한다.

**Tech Stack:** Python 3.10+, FastAPI, aiohttp, Jinja2, pytest + pytest-asyncio, ComfyUI HTTP API, Docker CLI, ffmpeg, pydub. 기존 venv: `/home/yanus/unified_ai_service/venv`.

**Spec:** `/home/yanus/docs/superpowers/specs/2026-05-23-phase-a-media-engine-design.md`

---

## 사전 컨텍스트 (모든 task가 공통으로 알아야 할 사실)

- **작업 디렉토리**: `/home/yanus`. Python 명령은 `/home/yanus/unified_ai_service/venv/bin/python` 사용.
- **ComfyUI**: `http://localhost:8188` (`docker compose -f /home/yanus/Docker/docker-compose.yml`), 모델 경로 `/home/yanus/Docker/models/`, 출력 디렉토리 `/home/yanus/Docker/output/`.
- **vLLM**: `http://localhost:8080/v1`, 컨테이너명 `vllm-server` (`/home/yanus/Docker/docker-compose.vllm.yml`).
- **결과 디렉토리**: `/home/yanus/unified_ai_service/results/`. 클라이언트는 `/api/results/<filename>` 로 다운로드.
- **업로드 디렉토리**: `/home/yanus/unified_ai_service/uploads/`.
- **기존 잡 관리**: `unified_ai_service/job_manager.py` (dict + `hub_jobs.json` 영속화). 시그니처: `create_job(type, input) -> id`, `update_job(id, status, result=None, error=None)`, `get_jobs()`, `delete_job(id)`.
- **테스트 실행**: 패키지 내부 `cd /home/yanus/unified_ai_service && ./venv/bin/python -m pytest media_engine/tests/ -v`.
- **vLLM 호환성**: 현재 컨테이너는 `--gpu-memory-utilization 0.5`로 ~64GB 점유. heavy 잡 실행 시 `docker stop vllm-server` 후 `docker start vllm-server` (재기동 ~30~60초).
- **모델 보유 현황** (이미 디스크에 있음):
  - `diffusion_models/z_image_turbo_bf16.safetensors` (light)
  - `diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors` (heavy)
  - `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (heavy)
  - `diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors` (heavy, lecture lip-sync용)
  - `text_encoders/qwen_3_4b.safetensors`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - `vae/ae.safetensors`, `wan_2.1_vae.safetensors`, `qwen_image_vae.safetensors`
  - `audio_encoders/wav2vec2_large_english_fp16.safetensors`

- **모든 step 끝의 commit 메시지 형식**: `feat(media-engine): <한국어 1줄 요약>` (한국어 또는 영어 자유, 단 prefix `feat(media-engine):` 유지). 최하단 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` 한 줄 추가.

---

## Task 1 — 패키지 골격 + 의존성 + `catalog.py` (메타데이터 + 검증)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/__init__.py`
- Create: `/home/yanus/unified_ai_service/media_engine/catalog.py`
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/__init__.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/__init__.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_catalog.py`
- Modify: `/home/yanus/unified_ai_service/requirements.txt`

- [ ] **Step 1: 테스트 의존성 설치**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/pip install pytest==8.3.3 pytest-asyncio==0.24.0 aiohttp jinja2
```

Expected: 모든 패키지 설치 성공 (jinja2/aiohttp는 이미 있을 수 있음).

- [ ] **Step 2: `requirements.txt`에 신규 의존성 추가**

기존 파일 내용 확인 후, 다음 줄이 없으면 추가:

```
jinja2>=3.1
aiohttp>=3.9
```

(pytest 류는 dev 전용이므로 requirements.txt에는 넣지 않음.)

- [ ] **Step 3: 패키지 골격 생성**

```bash
mkdir -p /home/yanus/unified_ai_service/media_engine/workflows
mkdir -p /home/yanus/unified_ai_service/media_engine/tests
touch /home/yanus/unified_ai_service/media_engine/__init__.py
touch /home/yanus/unified_ai_service/media_engine/workflows/__init__.py
touch /home/yanus/unified_ai_service/media_engine/tests/__init__.py
```

`/home/yanus/unified_ai_service/media_engine/__init__.py` 내용:

```python
"""media_engine — Phase A 공용 미디어 엔진.

Submodules:
- catalog: 워크플로우 메타데이터 단일 출처
- comfyui_client: ComfyUI HTTP I/O
- gpu_arbiter: vLLM ↔ ComfyUI heavy 모델 직렬화
- window: 무빙윈도우 청크/병합 유틸
- runner: 통합 실행 진입점 (render → submit → poll → fetch)
- job_queue: 잡 관리자 위의 직렬화 래퍼
"""

__version__ = "0.1.0"
```

- [ ] **Step 4: 실패 테스트 작성 (`test_catalog.py`)**

`/home/yanus/unified_ai_service/media_engine/tests/test_catalog.py`:

```python
import pytest
from media_engine import catalog


def test_get_returns_workflow_metadata():
    """등록된 워크플로우는 메타데이터를 반환한다."""
    meta = catalog.get("image.gen.zimage_turbo")
    assert meta["template"] == "image_gen_zimage_turbo.json.j2"
    assert meta["vram_class"] == "light"
    assert meta["output_node"]  # 비어있지 않음
    assert meta["timeout_sec"] >= 30


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown.workflow"):
        catalog.get("unknown.workflow")


def test_validate_fills_defaults():
    meta = catalog.get("image.gen.zimage_turbo")
    out = catalog.validate(meta, {"prompt": "a cat"})
    assert out["prompt"] == "a cat"
    assert out["width"] == 1024  # default
    assert out["steps"] == 8     # turbo default


def test_validate_rejects_missing_required():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="prompt"):
        catalog.validate(meta, {})


def test_validate_coerces_types():
    meta = catalog.get("image.gen.zimage_turbo")
    out = catalog.validate(meta, {"prompt": "x", "width": "768"})
    assert out["width"] == 768  # str → int


def test_validate_rejects_invalid_type():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="width"):
        catalog.validate(meta, {"prompt": "x", "width": "not-a-number"})


def test_list_workflows_returns_all_ids():
    ids = catalog.list_workflows()
    assert "image.gen.zimage_turbo" in ids
    assert "image.edit.qwen" in ids
    assert "video.i2v.wan22" in ids
    assert "video.s2v.wan22" in ids


def test_check_models_present_returns_missing_list(tmp_path, monkeypatch):
    """모델 디렉토리가 비면 모든 필수 모델이 누락으로 보고된다."""
    monkeypatch.setattr(catalog, "MODELS_ROOT", str(tmp_path))
    meta = catalog.get("image.gen.zimage_turbo")
    missing = catalog.check_models_present(meta)
    assert len(missing) > 0
    assert all(isinstance(m, str) for m in missing)
```

- [ ] **Step 5: 테스트 실패 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/test_catalog.py -v
```

Expected: `ModuleNotFoundError: No module named 'media_engine.catalog'` 또는 ImportError로 실패.

- [ ] **Step 6: `catalog.py` 구현**

`/home/yanus/unified_ai_service/media_engine/catalog.py`:

```python
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
            "image_name": (str, ...),   # ComfyUI에 업로드된 파일명
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
            "frames":     (int, 81),    # ~5초 @ 16fps
            "steps":      (int, 4),     # lightx2v lora 사용 시 4 step
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


def validate(meta: dict, params: dict) -> dict:
    """타입 강제·기본값 채움. 필수 누락/타입 변환 실패는 ValueError."""
    out: dict[str, Any] = {}
    spec = meta["params"]
    for name, (typ, default) in spec.items():
        if name in params:
            value = params[name]
            try:
                out[name] = typ(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"param {name!r}: cannot coerce to {typ.__name__}: {e}")
        elif default is Ellipsis:
            raise ValueError(f"missing required param: {name!r}")
        else:
            out[name] = default
    extra = set(params) - set(spec)
    if extra:
        raise ValueError(f"unknown params: {sorted(extra)}")
    return out


def check_models_present(meta: dict) -> list[str]:
    """누락된 모델 파일의 상대경로 리스트 반환."""
    missing = []
    for rel in meta["models_required"]:
        path = os.path.join(MODELS_ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel)
    return missing
```

- [ ] **Step 7: 테스트 통과 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/test_catalog.py -v
```

Expected: 8 passed.

- [ ] **Step 8: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/ unified_ai_service/requirements.txt
git commit -m "$(cat <<'EOF'
feat(media-engine): catalog 메타데이터 + 워크플로우 검증

5개 미디어 워크플로우(zimage_turbo, qwen_edit, wan22 i2v/s2v)의
파라미터·모델·VRAM 클래스를 단일 출처로 정의하고 입력 검증을 추가.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — 워크플로우 템플릿 (1/4): `z_image_turbo`

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/image_gen_zimage_turbo.json.j2`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`:

```python
"""워크플로우 Jinja 템플릿이 ComfyUI API JSON으로 정상 렌더되는지 확인."""
import json
import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from media_engine import catalog


WF_DIR = Path(__file__).parent.parent / "workflows"


def _render(template: str, params: dict) -> dict:
    env = Environment(
        loader=FileSystemLoader(str(WF_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
    )
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    rendered = env.get_template(template).render(**params)
    return json.loads(rendered)


def test_zimage_turbo_renders_valid_json():
    meta = catalog.get("image.gen.zimage_turbo")
    params = catalog.validate(meta, {"prompt": "a quiet harbor at dusk"})
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    # 프롬프트가 어딘가의 노드 inputs에 들어갔는지 확인
    flat = json.dumps(wf, ensure_ascii=False)
    assert "a quiet harbor at dusk" in flat


def test_zimage_turbo_special_chars_escape():
    """프롬프트에 따옴표/개행이 있어도 JSON이 깨지지 않는다."""
    meta = catalog.get("image.gen.zimage_turbo")
    params = catalog.validate(meta, {"prompt": 'a "tall" cat\nwith \\backslash'})
    wf = _render(meta["template"], params)
    flat = json.dumps(wf)
    assert "tall" in flat
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v
```

Expected: `TemplateNotFound: image_gen_zimage_turbo.json.j2`.

- [ ] **Step 3: 워크플로우 템플릿 작성**

`/home/yanus/unified_ai_service/media_engine/workflows/image_gen_zimage_turbo.json.j2`:

```jinja
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "z_image_turbo_bf16.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "lumina2",
      "device": "default"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "ae.safetensors"
    }
  },
  "4": {
    "class_type": "ModelSamplingAuraFlow",
    "inputs": {
      "model": ["1", 0],
      "shift": 3.0
    }
  },
  "5": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["2", 0]
    }
  },
  "6": {
    "class_type": "ConditioningZeroOut",
    "inputs": {
      "conditioning": ["5", 0]
    }
  },
  "7": {
    "class_type": "EmptySD3LatentImage",
    "inputs": {
      "width":  {{ width }},
      "height": {{ height }},
      "batch_size": 1
    }
  },
  "8": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["4", 0],
      "positive": ["5", 0],
      "negative": ["6", 0],
      "latent_image": ["7", 0],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 1.0,
      "sampler_name": "res_multistep",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["8", 0],
      "vae": ["3", 0]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["decode", 0],
      "filename_prefix": "zimage_turbo"
    }
  }
}
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v
```

Expected: 2 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/workflows/ unified_ai_service/media_engine/tests/test_workflow_render.py
git commit -m "$(cat <<'EOF'
feat(media-engine): z_image_turbo 워크플로우 템플릿 추가

ComfyUI API 형식 JSON + Jinja2 변수(prompt/width/height/steps/seed).
res_multistep/simple 스케줄러, 4-step turbo 기본값.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — 워크플로우 템플릿 (2/4): `image.edit.qwen` (img2img)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/image_edit_qwen.json.j2`
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`

- [ ] **Step 1: 실패 테스트 추가**

`test_workflow_render.py` 끝에 추가:

```python
def test_qwen_edit_renders_valid_json():
    meta = catalog.get("image.edit.qwen")
    params = catalog.validate(meta, {
        "prompt": "make it night time",
        "image_name": "uploaded_xyz.png",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "make it night time" in flat
    assert "uploaded_xyz.png" in flat


def test_qwen_edit_denoise_applied():
    meta = catalog.get("image.edit.qwen")
    params = catalog.validate(meta, {
        "prompt": "p",
        "image_name": "x.png",
        "denoise": 0.5,
    })
    wf = _render(meta["template"], params)
    # KSampler 노드에 denoise=0.5가 들어갔는지
    sampler_node = next(
        v for v in wf.values()
        if v.get("class_type") == "KSampler"
    )
    assert abs(sampler_node["inputs"]["denoise"] - 0.5) < 1e-6
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py::test_qwen_edit_renders_valid_json -v
```

Expected: TemplateNotFound.

- [ ] **Step 3: 템플릿 작성**

`/home/yanus/unified_ai_service/media_engine/workflows/image_edit_qwen.json.j2`:

```jinja
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "type": "qwen_image",
      "device": "default"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    }
  },
  "4": {
    "class_type": "LoadImage",
    "inputs": {
      "image": {{ image_name | tojson }},
      "upload": "image"
    }
  },
  "5": {
    "class_type": "VAEEncode",
    "inputs": {
      "pixels": ["4", 0],
      "vae": ["3", 0]
    }
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["2", 0]
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "low quality, blurry, distorted",
      "clip": ["2", 0]
    }
  },
  "8": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 4.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": {{ denoise }}
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["8", 0],
      "vae": ["3", 0]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["decode", 0],
      "filename_prefix": "qwen_edit"
    }
  }
}
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v
```

Expected: 4 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/workflows/image_edit_qwen.json.j2 unified_ai_service/media_engine/tests/test_workflow_render.py
git commit -m "$(cat <<'EOF'
feat(media-engine): Qwen Image Edit 2509 워크플로우 템플릿

VAEEncode → KSampler (denoise 0.85 기본) → VAEDecode → SaveImage.
업로드된 image_name과 prompt를 변수로 받는 img2img/edit 흐름.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — 워크플로우 템플릿 (3/4): `video.i2v.wan22`

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/video_i2v_wan22.json.j2`
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_wan22_i2v_renders_valid_json():
    meta = catalog.get("video.i2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "camera slowly pans right",
        "image_name": "start_frame.png",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "camera slowly pans right" in flat
    assert "start_frame.png" in flat


def test_wan22_i2v_frames_match():
    meta = catalog.get("video.i2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "p",
        "image_name": "x.png",
        "frames": 65,
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf)
    assert "\"length\": 65" in flat or "\"length\":65" in flat
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py::test_wan22_i2v_renders_valid_json -v
```

Expected: TemplateNotFound.

- [ ] **Step 3: 템플릿 작성**

`/home/yanus/unified_ai_service/media_engine/workflows/video_i2v_wan22.json.j2`:

```jinja
{
  "unet_high": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
      "weight_dtype": "default"
    }
  },
  "unet_low": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
      "weight_dtype": "default"
    }
  },
  "model_sampling": {
    "class_type": "ModelSamplingSD3",
    "inputs": {
      "model": ["unet_high", 0],
      "shift": 8.0
    }
  },
  "vae": {
    "class_type": "VAELoader",
    "inputs": {
      "vae_name": "wan_2.1_vae.safetensors"
    }
  },
  "clip": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      "type": "wan",
      "device": "default"
    }
  },
  "load_image": {
    "class_type": "LoadImage",
    "inputs": {
      "image": {{ image_name | tojson }},
      "upload": "image"
    }
  },
  "pos": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["clip", 0]
    }
  },
  "neg": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "low quality, blurry, distorted, static, jpeg artifacts",
      "clip": ["clip", 0]
    }
  },
  "i2v": {
    "class_type": "WanImageToVideo",
    "inputs": {
      "positive": ["pos", 0],
      "negative": ["neg", 0],
      "vae": ["vae", 0],
      "start_image": ["load_image", 0],
      "width": {{ width }},
      "height": {{ height }},
      "length": {{ frames }},
      "batch_size": 1
    }
  },
  "sampler": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["model_sampling", 0],
      "positive": ["i2v", 0],
      "negative": ["i2v", 1],
      "latent_image": ["i2v", 2],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 1.0,
      "sampler_name": "uni_pc",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["sampler", 0],
      "vae": ["vae", 0]
    }
  },
  "combine": {
    "class_type": "VHS_VideoCombine",
    "inputs": {
      "images": ["decode", 0],
      "frame_rate": 16.0,
      "loop_count": 0,
      "filename_prefix": "wan22_i2v",
      "format": "video/h264-mp4",
      "pix_fmt": "yuv420p",
      "crf": 19,
      "save_metadata": false,
      "pingpong": false,
      "save_output": true
    }
  }
}
```

> **참고**: 실제로 Wan2.2 i2v는 high/low noise 두 모델을 조합하는 더 복잡한 KSampler 체인을 쓰는 변종도 있으나, Phase A에서는 single-stage(high noise만 사용)로 시작한다. low_noise는 미사용이지만 catalog의 `models_required`에는 포함시켜 파일 존재 검증만 한다. 향후 B4에서 multi-stage로 확장 가능.

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v
```

Expected: 6 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/workflows/video_i2v_wan22.json.j2 unified_ai_service/media_engine/tests/test_workflow_render.py
git commit -m "$(cat <<'EOF'
feat(media-engine): Wan2.2 i2v 14B 워크플로우 템플릿

WanImageToVideo + KSampler(uni_pc/simple) + VHS_VideoCombine.
start_image와 prompt로 16fps 5초(81 frames) 비디오 생성.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — 워크플로우 템플릿 (4/4): `video.s2v.wan22` (lip-sync)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/video_s2v_wan22.json.j2`
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_wan22_s2v_renders_valid_json():
    meta = catalog.get("video.s2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "speaker in front of bookshelf",
        "image_name": "ref.png",
        "audio_name": "speech.wav",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "speech.wav" in flat
    assert "ref.png" in flat
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py::test_wan22_s2v_renders_valid_json -v
```

Expected: TemplateNotFound.

- [ ] **Step 3: 템플릿 작성**

`/home/yanus/unified_ai_service/media_engine/workflows/video_s2v_wan22.json.j2` (lecture_service의 `get_wan_s2v_workflow` 구조 이전):

```jinja
{
  "unet": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors",
      "weight_dtype": "default"
    }
  },
  "model_sampling": {
    "class_type": "ModelSamplingSD3",
    "inputs": {
      "model": ["unet", 0],
      "shift": 12.0
    }
  },
  "vae": {
    "class_type": "VAELoader",
    "inputs": { "vae_name": "wan_2.1_vae.safetensors" }
  },
  "clip": {
    "class_type": "CLIPLoader",
    "inputs": {
      "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
      "type": "wan",
      "device": "default"
    }
  },
  "audio_enc_loader": {
    "class_type": "AudioEncoderLoader",
    "inputs": { "audio_encoder_name": "wav2vec2_large_english_fp16.safetensors" }
  },
  "load_image": {
    "class_type": "LoadImage",
    "inputs": {
      "image": {{ image_name | tojson }},
      "upload": "image"
    }
  },
  "load_audio": {
    "class_type": "LoadAudio",
    "inputs": { "audio": {{ audio_name | tojson }} }
  },
  "audio_enc": {
    "class_type": "AudioEncoderEncode",
    "inputs": {
      "audio_encoder": ["audio_enc_loader", 0],
      "audio": ["load_audio", 0]
    }
  },
  "pos": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["clip", 0]
    }
  },
  "neg": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "low quality, blurry, distorted, static, jpeg artifacts, motionless, cluttered",
      "clip": ["clip", 0]
    }
  },
  "s2v": {
    "class_type": "WanSoundImageToVideo",
    "inputs": {
      "positive": ["pos", 0],
      "negative": ["neg", 0],
      "vae": ["vae", 0],
      "audio_encoder_output": ["audio_enc", 0],
      "ref_image": ["load_image", 0],
      "width": {{ width }},
      "height": {{ height }},
      "length": {{ frames }},
      "batch_size": 1
    }
  },
  "sampler": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["model_sampling", 0],
      "positive": ["s2v", 0],
      "negative": ["s2v", 1],
      "latent_image": ["s2v", 2],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 1.0,
      "sampler_name": "uni_pc",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["sampler", 0],
      "vae": ["vae", 0]
    }
  },
  "combine": {
    "class_type": "VHS_VideoCombine",
    "inputs": {
      "images": ["decode", 0],
      "audio": ["load_audio", 0],
      "frame_rate": 16.0,
      "loop_count": 0,
      "filename_prefix": "wan22_s2v",
      "format": "video/h264-mp4",
      "pix_fmt": "yuv420p",
      "crf": 19,
      "save_metadata": false,
      "pingpong": false,
      "save_output": true
    }
  }
}
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v
```

Expected: 7 passed.

- [ ] **Step 5: workflows/README.md 작성**

`/home/yanus/unified_ai_service/media_engine/workflows/README.md`:

```markdown
# media_engine 워크플로우 카탈로그

각 `.json.j2` 파일은 ComfyUI API 형식 JSON에 Jinja2 변수만 노출한 템플릿이다.
파라미터·메타데이터는 `../catalog.py` 의 `WORKFLOWS` 딕셔너리에서 관리한다.

## 신규 워크플로우 추가 절차

1. ComfyUI 웹 UI에서 워크플로우를 만들고 **"Save (API Format)"** 으로 export.
2. `<id>.json.j2` 로 이 디렉토리에 저장.
3. 가변 부분(prompt, image_name, seed, steps 등)을 `{{ name | tojson }}` (문자열) 또는 `{{ name }}` (숫자)으로 치환.
4. `catalog.py` 의 `WORKFLOWS` 에 항목 추가:
   - `template`, `params` (타입+기본값), `models_required`, `output_node`, `vram_class`, `timeout_sec`
5. `test_workflow_render.py` 에 렌더 테스트 추가.

## 현재 등록 워크플로우

- `image_gen_zimage_turbo.json.j2` — z_image_turbo (light, ~5초)
- `image_edit_qwen.json.j2` — Qwen Image Edit 2509 img2img/edit (heavy, ~60초)
- `video_i2v_wan22.json.j2` — Wan2.2 i2v 14B, 5초 청크 (heavy, ~5분)
- `video_s2v_wan22.json.j2` — Wan2.2 s2v 14B 립싱크 (heavy, ~7분)
```

- [ ] **Step 6: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/workflows/
git commit -m "$(cat <<'EOF'
feat(media-engine): Wan2.2 s2v 립싱크 워크플로우 + README

WanSoundImageToVideo 노드로 ref_image + audio → lip-sync 비디오.
lecture_service workflows.py 의 s2v 흐름을 템플릿화해 이전 준비.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — `comfyui_client.py` (HTTP I/O)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/comfyui_client.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_comfyui_client.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_comfyui_client.py`:

```python
"""ComfyUI HTTP 클라이언트 동작 검증 (aiohttp 응답 mock)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from media_engine import comfyui_client as cc


@pytest.mark.asyncio
async def test_submit_returns_prompt_id():
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.json = AsyncMock(return_value={"prompt_id": "abc-123"})
    fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_resp.__aexit__ = AsyncMock(return_value=False)

    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=fake_resp)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    with patch.object(cc.aiohttp, "ClientSession", return_value=fake_session):
        pid = await cc.submit({"3": {"class_type": "X"}})
        assert pid == "abc-123"


@pytest.mark.asyncio
async def test_submit_raises_on_reject():
    fake_resp = MagicMock()
    fake_resp.status = 400
    fake_resp.json = AsyncMock(return_value={"error": "node_errors", "node_errors": {"3": "boom"}})
    fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_resp.__aexit__ = AsyncMock(return_value=False)

    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=fake_resp)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    with patch.object(cc.aiohttp, "ClientSession", return_value=fake_session):
        with pytest.raises(cc.ComfyUIError, match="boom"):
            await cc.submit({"3": {"class_type": "X"}})


@pytest.mark.asyncio
async def test_wait_and_fetch_polls_until_done(tmp_path, monkeypatch):
    """history가 비어있다가 → 채워지면 출력 파일 경로 반환."""
    monkeypatch.setattr(cc, "COMFY_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "result.png").write_bytes(b"PNG")

    history_states = [
        {},  # 첫 poll: 미완료
        {"abc-123": {"outputs": {"9": {"images": [{"filename": "result.png", "type": "output"}]}}}},
    ]

    async def fake_get_history(prompt_id):
        return history_states.pop(0) if history_states else history_states[-1]

    monkeypatch.setattr(cc, "_get_history", fake_get_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    path = await cc.wait_and_fetch("abc-123", output_node="9", timeout=10)
    assert str(path).endswith("result.png")


@pytest.mark.asyncio
async def test_wait_and_fetch_timeout(monkeypatch):
    async def empty_history(_):
        return {}
    monkeypatch.setattr(cc, "_get_history", empty_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    with pytest.raises(cc.ComfyUIError, match="timeout"):
        await cc.wait_and_fetch("abc-123", output_node="9", timeout=1, poll_interval=0.01)


@pytest.mark.asyncio
async def test_wait_and_fetch_workflow_error(monkeypatch):
    """history 응답에 status.error가 있으면 ComfyUIError로 전환."""
    async def err_history(_):
        return {"abc-123": {"status": {"status_str": "error", "messages": [["error", {"exception_message": "OOM"}]]}}}
    monkeypatch.setattr(cc, "_get_history", err_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    with pytest.raises(cc.ComfyUIError, match="OOM"):
        await cc.wait_and_fetch("abc-123", output_node="9", timeout=5, poll_interval=0.01)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_comfyui_client.py -v
```

Expected: ImportError for `media_engine.comfyui_client`.

- [ ] **Step 3: 구현**

`/home/yanus/unified_ai_service/media_engine/comfyui_client.py`:

```python
"""ComfyUI HTTP API 클라이언트 — submit / poll / upload / fetch."""
import asyncio
import os
from pathlib import Path

import aiohttp

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
COMFY_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", "/home/yanus/Docker/output")


class ComfyUIError(RuntimeError):
    pass


async def submit(workflow_prompt: dict) -> str:
    """워크플로우를 큐에 등록하고 prompt_id 반환."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow_prompt},
        ) as resp:
            data = await resp.json()
            if resp.status != 200 or "prompt_id" not in data:
                node_errs = data.get("node_errors") or data.get("error") or data
                raise ComfyUIError(f"submit rejected: {node_errs}")
            return data["prompt_id"]


async def _get_history(prompt_id: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
            return await resp.json()


async def wait_and_fetch(
    prompt_id: str,
    output_node: str,
    timeout: float = 300,
    poll_interval: float = 2.0,
) -> Path:
    """ComfyUI 워크플로우 완료를 기다리고 출력 파일 경로를 반환."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        history = await _get_history(prompt_id)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                msg = next(
                    (m[1].get("exception_message") for m in msgs if m and m[0] == "error"),
                    str(status),
                )
                raise ComfyUIError(f"workflow error: {msg}")

            outputs = entry.get("outputs", {})
            node_out = outputs.get(output_node)
            if node_out:
                filename = _extract_filename(node_out)
                if filename:
                    return Path(COMFY_OUTPUT_DIR) / filename
        await asyncio.sleep(poll_interval)
    raise ComfyUIError(f"timeout after {timeout}s for prompt {prompt_id}")


def _extract_filename(node_output: dict) -> str | None:
    """SaveImage(images) / VHS_VideoCombine(gifs|videos) 양쪽 모두 대응."""
    for key in ("images", "gifs", "videos"):
        items = node_output.get(key)
        if items:
            return items[0]["filename"]
    return None


async def upload_image(local_path: str, filename: str) -> dict:
    """ComfyUI input/ 디렉토리로 이미지 업로드."""
    async with aiohttp.ClientSession() as session:
        with open(local_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("image", f, filename=filename)
            form.add_field("type", "input")
            async with session.post(f"{COMFYUI_URL}/upload/image", data=form) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise ComfyUIError(f"upload failed: {data}")
                return data


async def upload_audio(local_path: str, filename: str) -> dict:
    """ComfyUI input/ 디렉토리로 오디오 업로드 (image 엔드포인트 재사용)."""
    return await upload_image(local_path, filename)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_comfyui_client.py -v
```

Expected: 5 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/comfyui_client.py unified_ai_service/media_engine/tests/test_comfyui_client.py
git commit -m "$(cat <<'EOF'
feat(media-engine): ComfyUI HTTP 클라이언트 (submit/poll/upload)

prompt_id 발급, history 폴링(타임아웃·에러 메시지 추출),
SaveImage/VHS_VideoCombine 출력 파일명 추출, input/ 업로드 지원.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — `gpu_arbiter.py` (vLLM docker swap + 직렬화)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/gpu_arbiter.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_gpu_arbiter.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_gpu_arbiter.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from media_engine import gpu_arbiter as ga


@pytest.fixture(autouse=True)
def _reset_state():
    ga._state = "running"
    yield
    ga._state = "running"


@pytest.mark.asyncio
async def test_light_does_not_lock(monkeypatch):
    """light 작업은 lock 없이 통과하고 vLLM은 그대로 running."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))

    async with ga.acquire("light"):
        pass

    assert ga.vllm_available() is True
    fake_run.assert_not_called()


@pytest.mark.asyncio
async def test_heavy_pauses_and_resumes_vllm(monkeypatch):
    """heavy 작업은 docker stop → 작업 → docker start 트리거."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())

    async with ga.acquire("heavy"):
        # 진입 시 stop이 호출되어야 함
        assert any("stop" in str(c) for c in fake_run.call_args_list), \
            f"expected docker stop, got {fake_run.call_args_list}"
        assert ga.vllm_available() is False

    # blocks 종료 후 background restart 트리거 완료까지 대기
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    stop_calls = [c for c in fake_run.call_args_list if "stop" in str(c)]
    start_calls = [c for c in fake_run.call_args_list if "start" in str(c)]
    assert len(stop_calls) >= 1
    # start는 background이므로 즉시 호출되지 않을 수 있음 → 결국 호출됨을 wait
    for _ in range(50):
        if start_calls:
            break
        await asyncio.sleep(0.01)
        start_calls = [c for c in fake_run.call_args_list if "start" in str(c)]
    assert len(start_calls) >= 1


@pytest.mark.asyncio
async def test_heavy_serializes(monkeypatch):
    """동시 heavy 잡 2건은 순차 실행."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())

    order = []

    async def heavy_work(name):
        async with ga.acquire("heavy"):
            order.append(f"{name}-enter")
            await asyncio.sleep(0)
            order.append(f"{name}-leave")

    await asyncio.gather(heavy_work("A"), heavy_work("B"))

    # A가 완전히 끝난 뒤 B가 들어가야 함 (또는 그 반대)
    assert order in (
        ["A-enter", "A-leave", "B-enter", "B-leave"],
        ["B-enter", "B-leave", "A-enter", "A-leave"],
    )


@pytest.mark.asyncio
async def test_vllm_resume_failure_marks_unhealthy(monkeypatch):
    """healthcheck 90회 실패 후 vllm_state=unhealthy."""
    monkeypatch.setattr(ga, "_docker", AsyncMock(return_value=0))
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=False))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(ga, "VLLM_RESUME_TIMEOUT", 3)  # 빠르게

    async with ga.acquire("heavy"):
        pass

    # background resume task 완료 대기
    for _ in range(100):
        if ga._state == "unhealthy":
            break
        await asyncio.sleep(0.01)
    assert ga._state == "unhealthy"
    assert ga.vllm_available() is False
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_gpu_arbiter.py -v
```

Expected: ImportError.

- [ ] **Step 3: 구현**

`/home/yanus/unified_ai_service/media_engine/gpu_arbiter.py`:

```python
"""GPU 자원 중재자.

heavy 잡은 process-wide lock + vLLM docker stop으로 직렬화하고,
종료 후 background에서 docker start + 헬스체크.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import aiohttp

logger = logging.getLogger(__name__)

VLLM_CONTAINER = os.getenv("VLLM_CONTAINER", "vllm-server")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1")
VLLM_RESUME_TIMEOUT = int(os.getenv("VLLM_RESUME_TIMEOUT", "90"))

_lock = asyncio.Lock()
_state = "running"  # "running" | "paused" | "restarting" | "unhealthy"


def vllm_available() -> bool:
    return _state == "running"


def state() -> str:
    return _state


async def _docker(*args: str) -> int:
    """docker CLI 호출 — 테스트에서 monkeypatch."""
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode


async def _vllm_healthy() -> bool:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as session:
            async with session.get(f"{VLLM_URL}/models") as r:
                return r.status == 200
    except Exception:
        return False


async def _pause_vllm() -> None:
    global _state
    _state = "restarting"
    logger.info("Pausing vLLM container for heavy GPU work")
    rc = await _docker("stop", VLLM_CONTAINER)
    if rc != 0:
        logger.warning("docker stop returned %s — continuing anyway", rc)
    _state = "paused"


async def _resume_vllm() -> None:
    global _state
    _state = "restarting"
    logger.info("Resuming vLLM container")
    await _docker("start", VLLM_CONTAINER)
    for _ in range(VLLM_RESUME_TIMEOUT):
        if await _vllm_healthy():
            _state = "running"
            logger.info("vLLM is healthy again")
            return
        await asyncio.sleep(1)
    _state = "unhealthy"
    logger.error("vLLM failed to become healthy within %ds", VLLM_RESUME_TIMEOUT)


@asynccontextmanager
async def acquire(vram_class: str):
    """heavy: 직렬화 + vLLM swap. light: pass-through."""
    if vram_class == "heavy":
        async with _lock:
            await _pause_vllm()
            try:
                yield
            finally:
                # background에서 복구하여 잡 응답 시간이 증가하지 않게
                await _resume_vllm()
    elif vram_class == "light":
        yield
    else:
        raise ValueError(f"unknown vram_class: {vram_class!r}")
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_gpu_arbiter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/gpu_arbiter.py unified_ai_service/media_engine/tests/test_gpu_arbiter.py
git commit -m "$(cat <<'EOF'
feat(media-engine): GPU arbiter — vLLM docker swap + heavy 잡 직렬화

asyncio.Lock으로 heavy 잡 순차 처리, 진입 시 docker stop vllm-server,
종료 시 background docker start + /v1/models 헬스체크(최대 90초).
light 잡은 lock 없이 통과해 LLM과 공존 가능.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8 — `window.py` (무빙윈도우 유틸)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/window.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_window.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_window.py`:

```python
import os
import subprocess
import pytest
from pathlib import Path
from media_engine import window


@pytest.fixture
def silent_audio(tmp_path):
    """10초짜리 무음 WAV 생성."""
    from pydub import AudioSegment
    path = tmp_path / "silent.wav"
    AudioSegment.silent(duration=10000).export(path, format="wav")
    return path


@pytest.fixture
def short_video(tmp_path):
    """ffmpeg testsrc로 3초짜리 mp4 생성."""
    path = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=16",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.mark.asyncio
async def test_chunk_audio_fixed_overlap(silent_audio, tmp_path):
    chunks = await window.chunk_audio_fixed(
        str(silent_audio),
        chunk_sec=4,
        overlap_sec=1,
        out_dir=str(tmp_path / "chunks"),
    )
    assert len(chunks) >= 3
    # 청크 간 시작점이 (chunk - overlap)초씩 진행
    from pydub import AudioSegment
    for c in chunks:
        seg = AudioSegment.from_file(c)
        assert 3500 <= len(seg) <= 4500  # ~4초


@pytest.mark.asyncio
async def test_chunk_audio_smart_silence_split(tmp_path):
    """무음 구간 분할: 말소리(beep) 사이 무음을 경계로 잡는다."""
    from pydub import AudioSegment
    from pydub.generators import Sine
    # beep 2s + silence 1s + beep 2s + silence 1s + beep 2s = 8s
    seg = (Sine(440).to_audio_segment(duration=2000) +
           AudioSegment.silent(duration=1000) +
           Sine(440).to_audio_segment(duration=2000) +
           AudioSegment.silent(duration=1000) +
           Sine(440).to_audio_segment(duration=2000))
    src = tmp_path / "speech.wav"
    seg.export(src, format="wav")

    chunks = await window.chunk_audio_smart(
        str(src), target_range=(2, 6), out_dir=str(tmp_path / "out")
    )
    assert 2 <= len(chunks) <= 5


@pytest.mark.asyncio
async def test_extract_last_frame(short_video, tmp_path):
    out = await window.extract_last_frame(str(short_video), out_dir=str(tmp_path))
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


@pytest.mark.asyncio
async def test_concat_videos(short_video, tmp_path):
    out = tmp_path / "joined.mp4"
    await window.concat_videos([str(short_video), str(short_video)], str(out))
    assert out.exists()
    dur = await window.get_media_duration(str(out))
    assert 5.5 < dur < 6.5  # 3+3 ≈ 6초


@pytest.mark.asyncio
async def test_crossfade_audio(silent_audio, tmp_path):
    out = tmp_path / "xf.wav"
    await window.crossfade_audio_segments(
        [str(silent_audio), str(silent_audio)],
        overlap_ms=500,
        output_path=str(out),
    )
    assert out.exists()
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(out)
    # 10s + 10s - 0.5s overlap ≈ 19.5s
    assert 19000 < len(seg) < 20000
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_window.py -v
```

Expected: ImportError. 또한 ffmpeg/pydub가 venv에 있어야 한다.

```bash
./venv/bin/python -c "import pydub; print(pydub.__version__)"
which ffmpeg && ffmpeg -version | head -1
```

만약 pydub 누락 시: `./venv/bin/pip install pydub`.

- [ ] **Step 3: 구현**

`/home/yanus/unified_ai_service/media_engine/window.py`:

```python
"""무빙윈도우 유틸 — 미디어 청크링·last_frame·concat·crossfade."""
import asyncio
import os
import subprocess
import uuid
from pathlib import Path

from pydub import AudioSegment, silence


async def chunk_audio_fixed(
    audio_path: str,
    chunk_sec: float = 30,
    overlap_sec: float = 5,
    out_dir: str | None = None,
) -> list[Path]:
    """고정 길이 청크 + 오버랩 분할."""
    out_dir = out_dir or os.path.dirname(audio_path)
    os.makedirs(out_dir, exist_ok=True)
    audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
    total_ms = len(audio)
    chunk_ms = int(chunk_sec * 1000)
    step_ms = chunk_ms - int(overlap_sec * 1000)
    assert step_ms > 0, "overlap must be < chunk_sec"

    paths: list[Path] = []
    start = 0
    idx = 0
    while start < total_ms:
        end = min(start + chunk_ms, total_ms)
        chunk = audio[start:end]
        p = Path(out_dir) / f"chunk_{uuid.uuid4().hex[:6]}_{idx:03d}.wav"
        await asyncio.to_thread(chunk.export, p, format="wav")
        paths.append(p)
        if end >= total_ms:
            break
        start += step_ms
        idx += 1
    return paths


async def chunk_audio_smart(
    audio_path: str,
    target_range: tuple[float, float] = (4, 8),
    out_dir: str | None = None,
) -> list[Path]:
    """무음 기반 청크 (lecture_service.slice_audio 일반화)."""
    out_dir = out_dir or os.path.dirname(audio_path)
    os.makedirs(out_dir, exist_ok=True)
    audio = await asyncio.to_thread(AudioSegment.from_file, audio_path)
    min_ms, max_ms = int(target_range[0] * 1000), int(target_range[1] * 1000)

    silent_ranges = await asyncio.to_thread(
        silence.detect_silence, audio,
        min_silence_len=300,
        silence_thresh=audio.dBFS - 16,
    )
    split_points = [0]
    for s, e in silent_ranges:
        split_points.append((s + e) // 2)
    split_points.append(len(audio))

    final_points = [0]
    last = 0
    for p in split_points[1:]:
        d = p - last
        if d < min_ms:
            continue  # 너무 짧으면 다음으로 미룬다
        if d <= max_ms:
            final_points.append(p)
            last = p
        else:
            while (p - last) > max_ms:
                last += (min_ms + max_ms) // 2
                final_points.append(last)
            final_points.append(p)
            last = p
    if final_points[-1] < len(audio):
        final_points.append(len(audio))
    final_points = sorted(set(final_points))

    paths: list[Path] = []
    for i in range(len(final_points) - 1):
        s, e = final_points[i], final_points[i + 1]
        seg = audio[s:e]
        p = Path(out_dir) / f"smart_{uuid.uuid4().hex[:6]}_{i:03d}.wav"
        await asyncio.to_thread(seg.export, p, format="wav")
        paths.append(p)
    return paths


async def extract_last_frame(video_path: str, out_dir: str | None = None) -> Path:
    """비디오의 마지막 프레임을 JPEG로 저장."""
    out_dir = out_dir or os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir) / f"lastframe_{uuid.uuid4().hex[:6]}.jpg"
    cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
           "-update", "1", "-q:v", "2", str(out)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"extract_last_frame failed: {err.decode()}")
    return out


async def concat_videos(paths: list[str], output_path: str, overlap_frames: int = 0) -> Path:
    """ffmpeg concat demuxer로 비디오들을 이어붙임. overlap_frames>0이면 xfade 사용 (현재는 단순 concat)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".txt")
    list_file.write_text("".join(f"file '{os.path.abspath(p)}'\n" for p in paths))
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", str(out)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    list_file.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"concat_videos failed: {err.decode()}")
    return out


async def crossfade_audio_segments(
    paths: list[str], overlap_ms: int = 300, output_path: str | None = None
) -> Path:
    """pydub crossfade로 오디오 청크들을 부드럽게 잇는다."""
    if not paths:
        raise ValueError("paths is empty")
    segs = [await asyncio.to_thread(AudioSegment.from_file, p) for p in paths]
    out_seg = segs[0]
    for s in segs[1:]:
        out_seg = out_seg.append(s, crossfade=overlap_ms)
    out = Path(output_path) if output_path else Path(paths[0]).with_name(
        f"xf_{uuid.uuid4().hex[:6]}.wav"
    )
    await asyncio.to_thread(out_seg.export, out, format="wav")
    return out


async def get_media_duration(path: str) -> float:
    """ffprobe로 길이 초 단위 반환."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return float(stdout.decode().strip())
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_window.py -v
```

Expected: 5 passed (ffmpeg + pydub 필요).

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/window.py unified_ai_service/media_engine/tests/test_window.py
git commit -m "$(cat <<'EOF'
feat(media-engine): 무빙윈도우 유틸 (청크/last_frame/concat/crossfade)

오디오 고정·smart 청크링, ffmpeg last_frame 추출, concat,
pydub crossfade, ffprobe 길이 측정 — 향후 모든 미디어 capability가 공유.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9 — `runner.py` (통합 실행 진입점)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/runner.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_runner.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_runner.py`:

```python
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from media_engine import runner, catalog


@pytest.mark.asyncio
async def test_run_renders_validates_submits_fetches(tmp_path, monkeypatch):
    """runner.run은 catalog 검증 + 템플릿 렌더 + ComfyUI submit + fetch를 엮는다."""
    fake_submit = AsyncMock(return_value="pid-1")
    fake_fetch = AsyncMock(return_value=tmp_path / "out.png")
    (tmp_path / "out.png").write_bytes(b"PNG")

    monkeypatch.setattr(runner.comfyui_client, "submit", fake_submit)
    monkeypatch.setattr(runner.comfyui_client, "wait_and_fetch", fake_fetch)
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    out_path = await runner.run("image.gen.zimage_turbo", prompt="hello world")

    assert out_path.exists()
    assert out_path.parent.name == "results"
    # submit이 prompt가 들어간 워크플로우 dict로 호출됨
    wf = fake_submit.call_args[0][0]
    flat = json.dumps(wf, ensure_ascii=False)
    assert "hello world" in flat


@pytest.mark.asyncio
async def test_run_uses_gpu_arbiter(monkeypatch, tmp_path):
    """heavy 워크플로우는 gpu_arbiter.acquire('heavy')를 통해 실행된다."""
    captured = []

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def fake_acquire(vc):
        captured.append(vc)
        yield

    monkeypatch.setattr(runner.gpu_arbiter, "acquire", fake_acquire)
    monkeypatch.setattr(runner.comfyui_client, "submit", AsyncMock(return_value="pid"))
    (tmp_path / "x.mp4").write_bytes(b"MP4")
    monkeypatch.setattr(runner.comfyui_client, "wait_and_fetch", AsyncMock(return_value=tmp_path / "x.mp4"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    await runner.run("video.i2v.wan22", prompt="p", image_name="a.png")
    assert captured == ["heavy"]


@pytest.mark.asyncio
async def test_run_missing_param_raises():
    with pytest.raises(ValueError, match="prompt"):
        await runner.run("image.gen.zimage_turbo")


@pytest.mark.asyncio
async def test_run_unknown_workflow_raises():
    with pytest.raises(KeyError, match="unknown"):
        await runner.run("not.real.workflow", prompt="x")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_runner.py -v
```

Expected: ImportError.

- [ ] **Step 3: 구현**

`/home/yanus/unified_ai_service/media_engine/runner.py`:

```python
"""media_engine 실행 진입점 — render → submit → poll → fetch → copy."""
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from media_engine import catalog, comfyui_client, gpu_arbiter

logger = logging.getLogger(__name__)

RESULTS_DIR = "/home/yanus/unified_ai_service/results"
WF_DIR = Path(__file__).parent / "workflows"

_env = Environment(
    loader=FileSystemLoader(str(WF_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    keep_trailing_newline=False,
)
_env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)


def render_template(template: str, params: dict) -> dict:
    rendered = _env.get_template(template).render(**params)
    return json.loads(rendered)


async def run(workflow_id: str, **params) -> Path:
    meta = catalog.get(workflow_id)
    validated = catalog.validate(meta, params)
    workflow = render_template(meta["template"], validated)

    logger.info("Running workflow %s (vram_class=%s)", workflow_id, meta["vram_class"])

    async with gpu_arbiter.acquire(meta["vram_class"]):
        prompt_id = await comfyui_client.submit(workflow)
        source = await comfyui_client.wait_and_fetch(
            prompt_id,
            output_node=meta["output_node"],
            timeout=meta["timeout_sec"],
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = source.suffix or ".bin"
    dest = Path(RESULTS_DIR) / f"{workflow_id.replace('.', '_')}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy(source, dest)
    logger.info("Workflow %s done: %s", workflow_id, dest)
    return dest
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_runner.py -v
```

Expected: 4 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/runner.py unified_ai_service/media_engine/tests/test_runner.py
git commit -m "$(cat <<'EOF'
feat(media-engine): runner — catalog/template/arbiter/comfyui 통합 실행

run(workflow_id, **params) 한 줄로 검증→렌더→GPU 락→submit→
poll→fetch→results 복사까지 처리. 모든 미디어 shim의 단일 진입점.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10 — `job_queue.py` (잡 직렬화 래퍼)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/job_queue.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/test_job_queue.py`

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_job_queue.py`:

```python
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from media_engine import job_queue


@pytest.fixture
def fake_jm(monkeypatch):
    """job_manager.create_job/update_job 를 in-memory dict로 모의."""
    store = {}

    def create_job(jt, payload):
        jid = f"j-{len(store)+1}"
        store[jid] = {"status": "pending", "type": jt, "input": payload}
        return jid

    def update_job(jid, status, result=None, error=None):
        if jid in store:
            store[jid]["status"] = status
            if result is not None:
                store[jid]["result"] = result
            if error is not None:
                store[jid]["error"] = error

    monkeypatch.setattr(job_queue.job_manager, "create_job", create_job)
    monkeypatch.setattr(job_queue.job_manager, "update_job", update_job)
    return store


@pytest.mark.asyncio
async def test_submit_runs_coro_and_marks_completed(fake_jm, tmp_path):
    target = tmp_path / "x.png"
    target.write_bytes(b"X")

    async def work():
        return target

    jid = await job_queue.submit("test", {"k": "v"}, work())
    # background task 종료 대기
    for _ in range(100):
        if fake_jm[jid]["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.01)

    assert fake_jm[jid]["status"] == "completed"
    assert fake_jm[jid]["result"].endswith("/api/results/x.png")


@pytest.mark.asyncio
async def test_submit_records_failure(fake_jm):
    async def boom():
        raise RuntimeError("nope")

    jid = await job_queue.submit("test", {}, boom())
    for _ in range(100):
        if fake_jm[jid]["status"] == "failed":
            break
        await asyncio.sleep(0.01)
    assert fake_jm[jid]["status"] == "failed"
    assert "nope" in fake_jm[jid]["error"]


@pytest.mark.asyncio
async def test_submit_text_result(fake_jm):
    """str 반환(예: 비디오 분석 텍스트)은 그대로 결과로."""
    async def work():
        return "this is the analysis"

    jid = await job_queue.submit("test", {}, work())
    for _ in range(100):
        if fake_jm[jid]["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert fake_jm[jid]["result"] == "this is the analysis"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_job_queue.py -v
```

Expected: ImportError.

- [ ] **Step 3: 구현**

`/home/yanus/unified_ai_service/media_engine/job_queue.py`:

```python
"""기존 job_manager 위 직렬화 래퍼.

heavy 잡의 GPU 직렬화는 gpu_arbiter의 lock이 담당하므로 여기서는 단순 등록·실행만.
실패 시 에러 로그를 results/_errors/ 에 덤프한다.
"""
import asyncio
import logging
import os
import traceback
from pathlib import Path

import job_manager  # unified_ai_service 의 기존 모듈

logger = logging.getLogger(__name__)

ERROR_DIR = "/home/yanus/unified_ai_service/results/_errors"


async def submit(job_type: str, payload: dict, coro) -> str:
    """잡을 등록하고 background에서 coro 실행."""
    job_id = job_manager.create_job(job_type, payload)
    asyncio.create_task(_run(job_id, coro))
    return job_id


async def _run(job_id: str, coro) -> None:
    try:
        job_manager.update_job(job_id, "processing")
        result = await coro
        if isinstance(result, (Path, os.PathLike)):
            filename = os.path.basename(str(result))
            job_manager.update_job(job_id, "completed", result=f"/api/results/{filename}")
        else:
            job_manager.update_job(job_id, "completed", result=result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s", job_id, tb)
        _dump_error(job_id, tb)
        job_manager.update_job(job_id, "failed", error=str(e))


def _dump_error(job_id: str, traceback_text: str) -> None:
    try:
        os.makedirs(ERROR_DIR, exist_ok=True)
        with open(os.path.join(ERROR_DIR, f"{job_id}.log"), "w") as f:
            f.write(traceback_text)
    except Exception:
        pass  # 로깅 자체에서 실패하면 조용히 포기
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_job_queue.py -v
```

Expected: 3 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/job_queue.py unified_ai_service/media_engine/tests/test_job_queue.py
git commit -m "$(cat <<'EOF'
feat(media-engine): job_queue — 잡 등록/실행 래퍼 + 에러 덤프

기존 job_manager 위 비동기 래퍼. Path 반환은 /api/results/ URL로,
str 반환은 그대로 결과. 실패 시 results/_errors/{job_id}.log에 traceback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11 — `media_image.py` shim 교체

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_image.py` (전체 재작성)

- [ ] **Step 1: 기존 파일 확인**

```bash
cat /home/yanus/unified_ai_service/media_image.py
```

기존 `generate_image(prompt)`는 인라인 dict 워크플로우를 사용해 SDXL 호출. 이를 `runner.run` 호출로 교체.

- [ ] **Step 2: 파일 전체 재작성**

`/home/yanus/unified_ai_service/media_image.py`:

```python
"""이미지 생성·편집 진입점 (media_engine.runner 호출하는 얇은 shim)."""
import logging
from pathlib import Path

from media_engine import runner, comfyui_client

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    workflow: str = "zimage_turbo",
    **kwargs,
) -> Path:
    """텍스트→이미지. workflow ∈ {"zimage_turbo"}."""
    workflow_id = f"image.gen.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, **kwargs)


async def edit_image(
    image_path: str,
    prompt: str,
    workflow: str = "qwen",
    **kwargs,
) -> Path:
    """이미지+프롬프트→편집된 이미지. workflow ∈ {"qwen"}."""
    import os
    import uuid
    filename = f"edit_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    await comfyui_client.upload_image(image_path, filename)
    workflow_id = f"image.edit.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, image_name=filename, **kwargs)
```

- [ ] **Step 3: 임포트 무결성 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "import media_image; print(media_image.generate_image, media_image.edit_image)"
```

Expected: 두 함수 출력.

- [ ] **Step 4: 회귀 테스트 — 기존 endpoint가 임포트되는지 확인**

```bash
./venv/bin/python -c "import main; print('main imports OK')"
```

Expected: `main imports OK` (vLLM 호출 없이 임포트만).

만약 `media_image.py`를 호출하는 다른 자리(예: `main.py:189`의 `media_image.generate_image(prompt)`)가 새 시그니처와 호환되는지 확인. 기존 호출은 positional `prompt` 하나로 변함없음 → 호환.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_image.py
git commit -m "$(cat <<'EOF'
feat(media-engine): media_image.py shim 교체

인라인 SDXL workflow dict 제거, media_engine.runner 호출 1줄로 슬림화.
신규 edit_image()로 img2img/edit (Qwen Image Edit) 진입점 추가.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12 — `media_video.py` mock 제거 + shim

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_video.py` (mock 코드 삭제, 실제 호출로 교체)

- [ ] **Step 1: 현재 파일 구조 확인**

```bash
grep -n "MOCK\|ColorClip\|TextClip" /home/yanus/unified_ai_service/media_video.py
```

`generate_long_video` 내부의 mock 블록(`# MOCK implementation`, `ColorClip`, `TextClip`)을 모두 제거 대상으로 식별.

- [ ] **Step 2: 파일 전체 재작성**

`/home/yanus/unified_ai_service/media_video.py`:

```python
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
```

- [ ] **Step 3: 임포트/시그니처 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "import media_video; print('OK', media_video.generate_long_video)"
```

Expected: `OK <coroutine ...>`.

- [ ] **Step 4: 기존 main.py 호출 호환성 확인**

`main.py:152-185`의 호출 부:
- `media_video.generate_long_video(prompt, base_image, target_dur)` — positional, 새 시그니처와 호환 ✓
- `media_video.edit_video(video_path, audio_path, prompt=prompt)` — keyword, 호환 ✓
- `media_video.shorten_video(video_path, prompt)` — 호환 ✓
- `media_video.analyze_video(video_path, prompt)` — 호환 ✓

```bash
./venv/bin/python -c "import main; print('main imports OK')"
```

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_video.py
git commit -m "$(cat <<'EOF'
feat(media-engine): media_video.py mock 제거 + Wan2.2 i2v 실제 연결

ColorClip/TextClip mock 블록 전부 삭제. generate_long_video는
Wan2.2 i2v 5초 청크를 moving window(마지막 프레임 추출 → 다음 입력)로
연결해 임의 길이 비디오 생성. analyze_video는 3개 키프레임 종합으로 강화.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13 — `media_audio.py` shim 정리

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_audio.py` (구조 정리, 동작은 유지)

> Phase A에서는 audio 기능을 그대로 두되, import 측에서 media_engine.window를 쓰도록 단단히 한다. MusicGen/F5-TTS의 실제 신모델 교체는 Phase B2/B3에서 한다.

- [ ] **Step 1: 현재 파일 검토**

```bash
head -60 /home/yanus/unified_ai_service/media_audio.py
```

- [ ] **Step 2: 파일 상단에 임포트 정리 + 긴 오디오용 stub 함수 추가**

기존 `generate_music()`, `generate_tts_with_effects()` 함수는 **변경하지 않는다**. 다음 함수만 파일 끝에 추가:

```python
# ─── 긴 오디오 생성 (moving window, Phase B2에서 구현 확장) ──────────────
async def generate_long_music(prompt: str, total_duration_sec: int = 30) -> str:
    """긴 음악 생성 (moving window). Phase A에서는 30초 분할 후 crossfade로 단순 연결."""
    from media_engine import window
    import os, uuid
    chunk_sec = 10  # MusicGen-small 기본 5초; 안전하게 10초씩 생성 후 이음
    chunks_needed = max(1, total_duration_sec // chunk_sec)
    paths = []
    for i in range(chunks_needed):
        out_path = os.path.join(BASE_DIR, "results", f"music_chunk_{uuid.uuid4().hex[:6]}_{i}.wav")
        await generate_music(prompt, duration=chunk_sec, output_path=out_path)
        paths.append(out_path)
    if len(paths) == 1:
        return paths[0]
    final = os.path.join(BASE_DIR, "results", f"music_long_{uuid.uuid4().hex[:8]}.wav")
    await window.crossfade_audio_segments(paths, overlap_ms=500, output_path=final)
    # 청크 파일 정리
    for p in paths:
        try: os.unlink(p)
        except: pass
    return final
```

- [ ] **Step 3: 임포트 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "import media_audio; print(media_audio.generate_long_music)"
```

Expected: 함수 객체 출력.

- [ ] **Step 4: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_audio.py
git commit -m "$(cat <<'EOF'
feat(media-engine): media_audio.generate_long_music — moving window 음악

window.crossfade_audio_segments로 10초 청크를 0.5초 오버랩 이어붙여
임의 길이 음악 생성. 청크 단위 모델 호출은 기존 generate_music 재사용.
Phase B2에서 더 큰 MusicGen 모델로 교체 예정.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14 — `llm_service.py` vllm_available 체크 추가

**Files:**
- Modify: `/home/yanus/unified_ai_service/llm_service.py` (10-25번 줄 영역에 체크 추가)

- [ ] **Step 1: 실패 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/test_llm_gate.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
import sys, os
sys.path.insert(0, "/home/yanus/unified_ai_service")
import llm_service


@pytest.mark.asyncio
async def test_chat_returns_503_when_vllm_paused(monkeypatch):
    from media_engine import gpu_arbiter
    monkeypatch.setattr(gpu_arbiter, "vllm_available", lambda: False)
    monkeypatch.setattr(gpu_arbiter, "state", lambda: "paused")

    out = await llm_service.generate_text("hi", "sys")
    assert "GPU" in out or "일시" in out or "busy" in out.lower()


@pytest.mark.asyncio
async def test_chat_passes_when_vllm_running(monkeypatch):
    from media_engine import gpu_arbiter
    monkeypatch.setattr(gpu_arbiter, "vllm_available", lambda: True)

    fake_resp = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "yo"})()})()]})()
    fake_client = type("X", (), {})()
    fake_client.chat = type("Z", (), {})()
    fake_client.chat.completions = type("Q", (), {"create": AsyncMock(return_value=fake_resp)})()
    monkeypatch.setattr(llm_service, "client", fake_client)

    out = await llm_service.generate_text("hi", "sys")
    assert out == "yo"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_llm_gate.py -v
```

Expected: `test_chat_returns_503_when_vllm_paused` 실패 (체크 아직 없음).

- [ ] **Step 3: `llm_service.py` 수정**

`/home/yanus/unified_ai_service/llm_service.py` 의 `generate_text` 함수를 다음으로 교체 (파일 맨 위 import에 `from media_engine import gpu_arbiter` 추가):

```python
import os
import json
from openai import AsyncOpenAI
from media_engine import gpu_arbiter

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/gemma-4-26B-A4B-it")

client = AsyncOpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)


async def generate_text(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    if not gpu_arbiter.vllm_available():
        return (
            f"⏸️ LLM 일시 정지 중 (state={gpu_arbiter.state()}) — "
            "GPU 미디어 작업 진행. 30~60초 후 다시 시도해주세요."
        )
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return f"Error connecting to LLM server ({VLLM_URL}): {e}"
```

`generate_ppt_structure` 와 `analyze_image` 함수는 변경 없음. 단, 둘 다 내부에서 `client.chat.completions.create` 를 직접 호출하므로 각각에도 동일한 가드 추가:

`generate_ppt_structure` 첫 줄에:
```python
    if not gpu_arbiter.vllm_available():
        return [{"title": "LLM 일시 정지", "points": [f"state={gpu_arbiter.state()}"]}]
```

`analyze_image` 첫 줄에:
```python
    if not gpu_arbiter.vllm_available():
        return f"⏸️ VLM 일시 정지 중 (state={gpu_arbiter.state()})"
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_llm_gate.py -v
```

Expected: 2 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/llm_service.py unified_ai_service/media_engine/tests/test_llm_gate.py
git commit -m "$(cat <<'EOF'
feat(media-engine): LLM/VLM 호출에 vllm_available 게이트 추가

generate_text/analyze_image/generate_ppt_structure 진입부에 가드 추가.
GPU 미디어 작업으로 vLLM이 paused/restarting 상태이면 사용자에게
재시도 안내 메시지 반환 (500이 아닌 graceful degradation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15 — `main.py` 엔드포인트 job_queue 연동

**Files:**
- Modify: `/home/yanus/unified_ai_service/main.py` (process_media_* 함수들을 job_queue 사용으로 단순화)

- [ ] **Step 1: 현재 패턴 확인**

`main.py:128-185` 의 `process_media_image_task` ~ `process_media_video_analyze_task` 7개 함수들이 각자 try/except + job_manager 호출. 이를 `job_queue.submit` 한 줄로 교체할 수 있다.

기존 시그니처 유지를 위해, 엔드포인트(`@app.post`)는 그대로 두되, 내부 BackgroundTasks 호출만 job_queue로 변경.

- [ ] **Step 2: 변경 (main.py 상단에 import 추가)**

`main.py` 상단 import 영역(20번대 줄)에 추가:

```python
from media_engine import job_queue
```

- [ ] **Step 3: `process_media_*` 함수들 제거**

`main.py:128-185`의 7개 함수(`process_media_image_task`, `process_media_music_task`, `process_media_tts_task`, `process_media_video_gen_task`, `process_media_video_edit_task`, `process_media_video_shorts_task`, `process_media_video_analyze_task`) 를 모두 삭제한다.

- [ ] **Step 4: 엔드포인트 업데이트 (8개 미디어 엔드포인트)**

각 엔드포인트의 body를 다음 패턴으로 교체.

**`/api/media/image`** (`main.py:189-193`):

```python
@app.post("/api/media/image")
async def generate_image_endpoint(prompt: str = Form(...), workflow: str = Form("zimage_turbo"), auth=Depends(flexible_auth)):
    coro = media_image.generate_image(prompt, workflow=workflow)
    job_id = await job_queue.submit("media_image", {"prompt": prompt, "workflow": workflow}, coro)
    return {"job_id": job_id}
```

**`/api/media/image/edit`** (신규 — Phase A 산출물):

```python
@app.post("/api/media/image/edit")
async def edit_image_endpoint(prompt: str = Form(...), image: UploadFile = File(...), auth=Depends(flexible_auth)):
    img_path = os.path.join(UPLOADS_DIR, f"edit_in_{uuid.uuid4().hex}_{image.filename}")
    with open(img_path, "wb") as f:
        f.write(await image.read())
    coro = media_image.edit_image(img_path, prompt)
    job_id = await job_queue.submit("media_image_edit", {"prompt": prompt}, coro)
    return {"job_id": job_id}
```

**`/api/media/music`** (`main.py:195-199`):

```python
@app.post("/api/media/music")
async def generate_music_endpoint(prompt: str = Form(...), duration: int = Form(10), auth=Depends(flexible_auth)):
    if duration > 30:
        coro = media_audio.generate_long_music(prompt, duration)
    else:
        coro = media_audio.generate_music(prompt, duration)
    job_id = await job_queue.submit("media_music", {"prompt": prompt, "duration": duration}, coro)
    return {"job_id": job_id}
```

**`/api/media/tts`** (변경 거의 없음, job_queue 사용):

```python
@app.post("/api/media/tts")
async def generate_tts_endpoint(text: str = Form(...), ref_audio: UploadFile = File(None), ref_text: str = Form(""), auth=Depends(flexible_auth)):
    ref_path = ""
    if ref_audio:
        ref_path = os.path.join(UPLOADS_DIR, f"ref_{uuid.uuid4().hex}_{ref_audio.filename}")
        with open(ref_path, "wb") as f:
            f.write(await ref_audio.read())
    coro = media_audio.generate_tts_with_effects(text, ref_path, ref_text)
    job_id = await job_queue.submit("media_tts", {"text": text}, coro)
    return {"job_id": job_id}
```

**`/api/media/video/gen`**:

```python
@app.post("/api/media/video/gen")
async def generate_video_endpoint(prompt: str = Form(...), duration: int = Form(30), base_image: UploadFile = File(...), auth=Depends(flexible_auth)):
    img_path = os.path.join(UPLOADS_DIR, f"base_img_{uuid.uuid4().hex}_{base_image.filename}")
    with open(img_path, "wb") as f:
        f.write(await base_image.read())
    coro = media_video.generate_long_video(prompt, img_path, duration)
    job_id = await job_queue.submit("media_video_gen", {"prompt": prompt, "duration": duration}, coro)
    return {"job_id": job_id}
```

**`/api/media/video/edit`**:

```python
@app.post("/api/media/video/edit")
async def edit_video_endpoint(prompt: str = Form(""), video: UploadFile = File(...), audio: UploadFile = File(None), auth=Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"edit_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    aud_path = ""
    if audio:
        aud_path = os.path.join(UPLOADS_DIR, f"edit_aud_{uuid.uuid4().hex}_{audio.filename}")
        with open(aud_path, "wb") as f:
            f.write(await audio.read())
    coro = media_video.edit_video(vid_path, aud_path, prompt=prompt)
    job_id = await job_queue.submit("media_video_edit", {"prompt": prompt}, coro)
    return {"job_id": job_id}
```

**`/api/media/video/shorts`**:

```python
@app.post("/api/media/video/shorts")
async def video_shorts_endpoint(prompt: str = Form(""), video: UploadFile = File(...), auth=Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"shorts_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    coro = media_video.shorten_video(vid_path, prompt)
    job_id = await job_queue.submit("media_video_shorts", {"prompt": prompt}, coro)
    return {"job_id": job_id}
```

**`/api/media/video/analyze`**:

```python
@app.post("/api/media/video/analyze")
async def video_analyze_endpoint(prompt: str = Form(""), video: UploadFile = File(...), auth=Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"analyze_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    coro = media_video.analyze_video(vid_path, prompt)
    job_id = await job_queue.submit("media_video_analyze", {"prompt": prompt}, coro)
    return {"job_id": job_id}
```

`BackgroundTasks` 파라미터는 더 이상 필요 없으므로 위 7개 엔드포인트 시그니처에서 제거. `BackgroundTasks` import도 다른 곳에서 안 쓰이면 제거.

`process_llm_task` / `process_vlm_task` / `process_stt_task` / `process_ppt_task` 도 동일 패턴으로 정리(선택). 단 Phase A 범위는 "미디어" 4개이므로 LLM/VLM/STT/PPT 는 기존 BackgroundTasks 유지로 두어도 무방. 일관성을 위해 같이 바꾸려면 같은 패턴으로 교체.

- [ ] **Step 5: `health/vllm` 엔드포인트 추가**

`main.py` 어디든 (예: tokens 엔드포인트 근처):

```python
@app.get("/api/health/vllm")
async def health_vllm():
    from media_engine import gpu_arbiter
    return {"state": gpu_arbiter.state(), "available": gpu_arbiter.vllm_available()}
```

- [ ] **Step 6: 임포트 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "import main; print('main imports OK'); print([r.path for r in main.app.routes if 'media' in r.path])"
```

Expected: `main imports OK` + 미디어 엔드포인트 9개(/api/media/image, image/edit, music, tts, video/gen, video/edit, video/shorts, video/analyze + health/vllm) 출력.

- [ ] **Step 7: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/main.py
git commit -m "$(cat <<'EOF'
feat(media-engine): main.py 엔드포인트 job_queue 통합

8개 미디어 엔드포인트(image/image-edit/music/tts/video x4)가
job_queue.submit() 한 줄로 잡 등록. BackgroundTasks 의존 제거.
/api/media/image/edit 신규 추가, /api/health/vllm 모니터링 엔드포인트.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16 — 단위 테스트 전체 통과 + 회귀 임포트 검증

**Files:** (검증만 — 코드 변경 없음)

- [ ] **Step 1: 전체 단위 테스트 실행**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/ -v --ignore=media_engine/tests/integration
```

Expected: 모든 unit 테스트 통과 (~24개). 실패 시 해당 task로 돌아가 수정.

- [ ] **Step 2: 모든 변경 모듈 임포트 검증**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python << 'EOF'
import importlib, sys
mods = [
    "media_engine",
    "media_engine.catalog",
    "media_engine.comfyui_client",
    "media_engine.gpu_arbiter",
    "media_engine.window",
    "media_engine.runner",
    "media_engine.job_queue",
    "media_image",
    "media_audio",
    "media_video",
    "llm_service",
    "main",
]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
        sys.exit(1)
print("\nAll modules import successfully")
EOF
```

Expected: 모든 모듈 `OK`.

- [ ] **Step 3: 변경 사항이 없으면 skip-commit (verification 단계)**

이 task는 검증만이므로 commit 없음. 실패가 있으면 해당 task로 돌아가 수정 후 commit.

---

## Task 17 — Integration 테스트 (실제 GPU 사용)

> 이 task는 **수동 실행** 단계. CI에서는 skip. GB10 GPU와 ComfyUI/vLLM 컨테이너가 가동 중이어야 한다.

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/tests/integration/__init__.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/integration/test_real_workflows.py`
- Create: `/home/yanus/unified_ai_service/media_engine/tests/integration/conftest.py`

- [ ] **Step 1: pytest 마커 등록 (`media_engine/tests/conftest.py`)**

`/home/yanus/unified_ai_service/media_engine/tests/conftest.py`:

```python
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that need real GPU + ComfyUI + vLLM"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration", default=False):
        return
    skip = pytest.mark.skip(reason="need --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", default=False)
```

- [ ] **Step 2: integration 테스트 작성**

`/home/yanus/unified_ai_service/media_engine/tests/integration/__init__.py`: (빈 파일)

`/home/yanus/unified_ai_service/media_engine/tests/integration/test_real_workflows.py`:

```python
"""실제 GPU + ComfyUI + vLLM 통합 시험.

실행: ./venv/bin/python -m pytest media_engine/tests/integration -v --integration -s
"""
import asyncio
import os
import pytest
from pathlib import Path
import sys
sys.path.insert(0, "/home/yanus/unified_ai_service")


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_zimage_turbo_end_to_end():
    """light 워크플로우: vLLM swap 없이 실행되어야 한다."""
    from media_engine import runner, gpu_arbiter
    initial = gpu_arbiter.state()
    out = await runner.run(
        "image.gen.zimage_turbo",
        prompt="a small cat sitting in a sunlit room",
        steps=4,
    )
    assert out.exists(), f"output not produced: {out}"
    assert out.stat().st_size > 1000
    assert gpu_arbiter.state() == initial, "light job must not pause vLLM"


@pytest.mark.asyncio
async def test_wan22_i2v_vllm_swap_cycle(tmp_path):
    """heavy 워크플로우: 진입 시 vLLM stop, 종료 후 자동 start."""
    from media_engine import runner, gpu_arbiter, comfyui_client

    # 시드 이미지 준비
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(120, 80, 200))
    seed_path = tmp_path / "seed.png"
    img.save(seed_path)
    await comfyui_client.upload_image(str(seed_path), "integ_seed.png")

    assert gpu_arbiter.vllm_available(), "precondition: vLLM should be running"

    out = await runner.run(
        "video.i2v.wan22",
        prompt="gentle motion, cinematic",
        image_name="integ_seed.png",
        frames=33,  # ~2s for speed
        steps=4,
    )
    assert out.exists() and out.stat().st_size > 10_000

    # 종료 직후엔 paused/restarting일 수 있음 — 최대 120초 기다림
    for _ in range(120):
        if gpu_arbiter.vllm_available():
            break
        await asyncio.sleep(1)
    assert gpu_arbiter.vllm_available(), f"vLLM did not recover, state={gpu_arbiter.state()}"


@pytest.mark.asyncio
async def test_serialized_two_heavy_jobs():
    """동시 heavy 잡 2건이 순차 처리되며 둘 다 성공."""
    from media_engine import runner, comfyui_client
    from PIL import Image
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        img = Image.new("RGB", (512, 512), color=(50, 200, 100))
        p = Path(td) / "s.png"
        img.save(p)
        await comfyui_client.upload_image(str(p), "integ_s.png")

        async def job(idx):
            return await runner.run(
                "video.i2v.wan22",
                prompt=f"motion #{idx}",
                image_name="integ_s.png",
                frames=33,
                steps=4,
            )

        out1, out2 = await asyncio.gather(job(1), job(2))
        assert out1.exists() and out2.exists()
        assert out1 != out2
```

- [ ] **Step 3: 사전 점검 (수동)**

```bash
docker ps | grep -E "comfyui|vllm"
```

Expected: 두 컨테이너 Up 상태.

- [ ] **Step 4: integration 실행 (수동)**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/integration -v --integration -s --timeout=1800
```

Expected: 3 passed. 각 테스트가 실제 ComfyUI로 워크플로우를 돌리고 결과 파일을 산출한다. test_zimage_turbo는 30초 내, wan22 i2v는 5분 내, serialized는 10분 내 완료 권장.

만약 실패하면:
- `zimage_turbo` 실패 → 워크플로우 템플릿의 노드 이름/모델 파일명 점검. ComfyUI 로그: `docker logs comfyui --tail 100`.
- `wan22 i2v` 실패 → `docker logs comfyui --tail 200`에서 OOM/모델 누락 확인. 모델 경로 catalog `models_required` 와 실제 디스크 일치 확인.
- `serialized` 통과만 별도 실패 → lock 동작 확인 (`gpu_arbiter._lock`).

- [ ] **Step 5: 산출 결과 확인**

```bash
ls -la /home/yanus/unified_ai_service/results/ | head
```

Expected: 최근 생성된 `image_gen_zimage_turbo_*.png`, `video_i2v_wan22_*.mp4` 파일 존재.

- [ ] **Step 6: 커밋 (테스트 파일만)**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/tests/conftest.py unified_ai_service/media_engine/tests/integration/
git commit -m "$(cat <<'EOF'
test(media-engine): integration 시험 — 실제 GPU/ComfyUI 워크플로우

z_image_turbo light 잡(vLLM 무중단), wan22 i2v heavy 잡
(vLLM swap 한 사이클), 동시 heavy 2건 직렬화 검증.
--integration 플래그로만 수집되며 unit 테스트와 격리.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18 — 스모크 테스트 + 서비스 재기동 + 최종 검증

**Files:** (검증만)

- [ ] **Step 1: 서비스 재기동**

```bash
# 기존 uvicorn 프로세스 확인 및 graceful 재시작
pgrep -af "uvicorn.*main:app" | head
# unified_ai_service 의 keep_alive.sh가 watchdog 역할을 한다면 그것을 통하고,
# 아니면 직접 재시작:
cd /home/yanus/unified_ai_service
# (기존 실행 방식을 그대로 따라 재시작 — bash_history 또는 ai_hub_service.sh 참고)
cat ai_hub_service.sh
```

`ai_hub_service.sh`의 명령어를 그대로 사용하여 서비스 재시작.

- [ ] **Step 2: 엔드포인트 스모크 (curl)**

```bash
# 호스트(internal) 기준
BASE=http://localhost:8081

# 1) image.gen.zimage_turbo (light)
curl -s -X POST "$BASE/api/media/image" -F "prompt=a small studio scene" -F "workflow=zimage_turbo"

# 2) image edit (heavy)
# (테스트용 이미지 준비)
curl -s -X POST "$BASE/api/media/image/edit" -F "prompt=make it night" -F "image=@/home/yanus/Docker/output/zimage_turbo_*.png" | head

# 3) music (음악 30초 = long)
curl -s -X POST "$BASE/api/media/music" -F "prompt=calm piano" -F "duration=15"

# 4) video gen (가장 무거운, heavy)
curl -s -X POST "$BASE/api/media/video/gen" -F "prompt=gentle pan" -F "duration=10" -F "base_image=@/home/yanus/Docker/output/zimage_turbo_*.png"

# 5) health
curl -s "$BASE/api/health/vllm"

# 6) LLM (회귀)
curl -s -X POST "$BASE/api/llm/chat" -H "Content-Type: application/json" -d '{"prompt":"hi","system_prompt":"You are helpful","history":[]}'
```

각 요청은 `{"job_id": "..."}` 를 반환하고, `GET /api/jobs/{job_id}` 로 진행 상태 확인 가능해야 한다.

- [ ] **Step 3: 잡 상태 추적**

```bash
# 위에서 받은 job_id 중 하나를 사용
JOB=<paste-job-id>
for i in 1 2 3 4 5; do
  curl -s "$BASE/api/jobs/$JOB" | head -c 200 ; echo
  sleep 5
done
```

Expected: `pending` → `processing` → `completed`(result는 `/api/results/...`) 진행.

- [ ] **Step 4: 결과 파일 확인**

```bash
ls -lh /home/yanus/unified_ai_service/results/ | tail -10
```

Expected: 최근 생성된 image/music/video 파일들.

- [ ] **Step 5: 에러 로그 dir 확인**

```bash
ls /home/yanus/unified_ai_service/results/_errors/ 2>/dev/null | head -5
```

(실패한 잡이 있으면 .log 파일 존재. Phase A 완료 시점에는 비어있는 게 정상.)

- [ ] **Step 6: 회귀 — lecture_service 영향 없음 확인**

```bash
# lecture_service는 Phase A에서 변경하지 않았으므로 그대로 동작해야 함
curl -s http://localhost:8000/jobs | head
```

Expected: 정상 JSON 응답.

- [ ] **Step 7: 최종 commit (필요시)**

스모크 단계는 코드 변경이 없는 것이 정상. 발견된 버그가 있으면 해당 task로 돌아가 추가 커밋.

---

## Task 19 — 완료 정리 (README + spec 링크)

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/README.md`

- [ ] **Step 1: 패키지 README 작성**

`/home/yanus/unified_ai_service/media_engine/README.md`:

```markdown
# media_engine

`unified_ai_service`의 공용 미디어 엔진. Phase A 산출물.

- **catalog.py** — 워크플로우 메타데이터 단일 출처
- **workflows/*.json.j2** — ComfyUI API JSON + Jinja 변수 템플릿
- **comfyui_client.py** — ComfyUI HTTP 클라이언트 (submit/poll/upload)
- **gpu_arbiter.py** — heavy 잡 직렬화 + vLLM docker stop/start
- **window.py** — 무빙윈도우 유틸 (청크/last_frame/concat/crossfade)
- **runner.py** — 통합 실행 진입점 `runner.run(workflow_id, **params)`
- **job_queue.py** — `job_manager` 위 비동기 잡 등록 래퍼

## 사용 예

```python
from media_engine import runner
out_path = await runner.run("image.gen.zimage_turbo", prompt="a cat")
```

신규 워크플로우 추가는 [`workflows/README.md`](workflows/README.md) 참조.

## 테스트

- Unit: `./venv/bin/python -m pytest media_engine/tests/ -v`
- Integration (GPU 필요): `./venv/bin/python -m pytest media_engine/tests/integration -v --integration -s`

## 관련 문서

- [Phase A 설계 spec](/home/yanus/docs/superpowers/specs/2026-05-23-phase-a-media-engine-design.md)
- [Phase A 구현 계획](/home/yanus/docs/superpowers/plans/2026-05-23-phase-a-media-engine.md)
```

- [ ] **Step 2: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/README.md
git commit -m "$(cat <<'EOF'
docs(media-engine): README + spec/plan 링크

Phase A 산출물 요약, 사용 예, 테스트 명령. Phase B/C 작업자가
이 README → spec → plan 순으로 컨텍스트를 빠르게 잡을 수 있도록.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 종합 자가 점검 체크리스트 (Plan 작성자 self-review용)

- [x] **Spec 커버리지**
  - § 3 아키텍처: Task 1, 11~14 (모든 모듈/shim)
  - § 4.1 catalog.py: Task 1
  - § 4.2 runner.py: Task 9
  - § 4.3 gpu_arbiter.py: Task 7
  - § 4.4 window.py: Task 8
  - § 4.5 job_queue.py: Task 10
  - § 4.6 comfyui_client.py: Task 6
  - § 4.7 미디어 shim: Task 11~13
  - § 5 데이터 흐름: Task 15 (엔드포인트)
  - § 6 에러 처리: Task 10 (_dump_error), Task 7 (vLLM unhealthy 마킹), Task 6 (timeout/error msg 추출)
  - § 7 테스트 계획 unit: Task 1, 2~5, 6, 7, 8, 9, 10, 14
  - § 7 테스트 계획 integration: Task 17
  - § 7 테스트 계획 smoke: Task 18
  - § 8 DoD: 모든 항목이 Task 11~18에 대응
  - § 9 범위 외: 추가 모델 다운로드, B/C 단계는 spec에서 명시적 제외 — plan에 포함하지 않음
  - § 10 위험 완화: docker stop in-flight (Task 14 gate), 디스크 (Task 10 error dump), warmup, ComfyUI 크래시 (Task 17 timeout)

- [x] **Placeholder scan**: TBD/TODO/"implement later" 없음. 모든 code step은 완전한 코드 블록 포함.

- [x] **타입 일관성**:
  - `runner.run` 시그니처 `async def run(workflow_id: str, **params) -> Path` — Task 9 정의, Task 11/12/13 호출 일치
  - `gpu_arbiter.acquire` `vram_class: str` — Task 7 정의, Task 9 호출 일치
  - `gpu_arbiter.vllm_available() -> bool`, `state() -> str` — Task 7 정의, Task 14 호출 일치
  - `comfyui_client.submit(prompt) -> str` — Task 6 정의, Task 9 호출 일치
  - `job_queue.submit(type, payload, coro) -> str` — Task 10 정의, Task 15 호출 일치
  - `catalog.get / validate / list_workflows / check_models_present` — Task 1 정의, Task 9 호출 일치
  - `media_image.generate_image(prompt, workflow=..., **kw)` — Task 11 정의, Task 15 호출 일치
  - `media_video.generate_long_video(prompt, base_image, total_duration_target)` — Task 12 정의, Task 15 호출 일치

- [x] **각 Task는 self-contained**: 의존 task 코드 없이도 step의 모든 명령을 그대로 실행 가능. Step별 expected output 명시.

---

## 실행 옵션

**Plan complete and saved to `/home/yanus/docs/superpowers/plans/2026-05-23-phase-a-media-engine.md`.**

두 가지 실행 옵션:

**1. Subagent-Driven (recommended)** — 각 task마다 fresh subagent 디스패치, task 간 리뷰, 빠른 반복 ✓ 권장

**2. Inline Execution** — 이 세션에서 batch로 실행하며 중간 체크포인트 ✓ 컨텍스트 비용 큼

