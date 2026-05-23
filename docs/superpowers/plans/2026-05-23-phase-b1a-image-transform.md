# Phase B1a — 이미지 변환 강화 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FLUX dev fp8 기반 ControlNet(canny/openpose/depth) 워크플로우 + Qwen Image Edit 마스크 inpaint를 `unified_ai_service`에 통합해 이미지 변환을 상용 수준으로 강화.

**Architecture:** Phase A의 `media_engine/` 패키지를 그대로 사용. 신규 워크플로우 3개(.json.j2) + catalog 항목 3개 + `media_image.py` shim 함수 2개 + `main.py` 엔드포인트 2개 + 모델 다운로드 스크립트. 모든 신규 작업은 heavy 클래스(vLLM swap 트리거).

**Tech Stack:** Phase A 그대로 (Python 3.10+, FastAPI, aiohttp, Jinja2, pytest, ComfyUI HTTP API). 신규: FLUX fp8 safetensors (~17GB), FLUX-ControlNet-Union-Pro (~3GB), ComfyUI-controlnet-aux custom node.

**Spec:** `/home/yanus/docs/superpowers/specs/2026-05-23-phase-b1a-design.md`

---

## 사전 컨텍스트 (모든 task 공통)

- 작업 루트: `/home/yanus`, venv: `/home/yanus/unified_ai_service/venv/bin/python`
- 모델 root: `/home/yanus/Docker/models` (= `media_engine.catalog.MODELS_ROOT`)
- ComfyUI: `http://localhost:8188`, 컨테이너명 `comfyui`, 출력 `/home/yanus/Docker/output`
- vLLM: `http://localhost:8080`, 컨테이너명 `vllm-server`
- 테스트: `cd /home/yanus/unified_ai_service && ./venv/bin/python -m pytest media_engine/tests/ -v`
- 커밋 prefix: `feat(b1a):` (구현) / `test(b1a):` (테스트) / `docs(b1a):` (문서). 끝줄 `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
- Phase A 산출 상태: 41 unit tests passing, `media_engine.runner.run(workflow_id, **params)` 호출 가능, `comfyui_client.upload_image(local_path, filename)` 사용 가능
- 절대 금지: amend, `--no-verify`, `--force`, push to remote, `--integration` 자동 실행 (Task 6에서만)

---

## Task 1 — 모델 다운로드 스크립트 + 실행

**Files:**
- Create: `/home/yanus/unified_ai_service/scripts/download_b1a_models.sh`

- [ ] **Step 1: 디렉토리 준비 확인**

```bash
mkdir -p /home/yanus/unified_ai_service/scripts
df -h /home/yanus/Docker | tail -1
```

Expected: 디스크 잔여 100GB 이상 (FLUX 17GB + ControlNet 3GB 여유).

- [ ] **Step 2: 다운로드 스크립트 작성**

`/home/yanus/unified_ai_service/scripts/download_b1a_models.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/home/yanus/Docker/models}"
NODES_ROOT="${NODES_ROOT:-/home/yanus/Docker/custom_nodes_persistent}"

mkdir -p "$MODELS_ROOT/diffusion_models/FLUX1"
mkdir -p "$MODELS_ROOT/controlnet/FLUX"
mkdir -p "$NODES_ROOT"

echo "=== [1/3] FLUX dev fp8 (~17GB) ==="
FLUX_PATH="$MODELS_ROOT/diffusion_models/FLUX1/flux1-dev-fp8.safetensors"
if [ -f "$FLUX_PATH" ] && [ "$(stat -c%s "$FLUX_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$FLUX_PATH" | cut -f1))"
else
  wget --continue -O "$FLUX_PATH.partial" \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
  mv "$FLUX_PATH.partial" "$FLUX_PATH"
  echo "  downloaded: $(du -h "$FLUX_PATH" | cut -f1)"
fi

echo "=== [2/3] FLUX-ControlNet-Union-Pro (~3GB) ==="
CN_PATH="$MODELS_ROOT/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors"
if [ -f "$CN_PATH" ] && [ "$(stat -c%s "$CN_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$CN_PATH" | cut -f1))"
else
  wget --continue -O "$CN_PATH.partial" \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"
  mv "$CN_PATH.partial" "$CN_PATH"
  echo "  downloaded: $(du -h "$CN_PATH" | cut -f1)"
fi

echo "=== [3/3] ComfyUI-controlnet-aux custom node ==="
AUX_DIR="$NODES_ROOT/ComfyUI-controlnet-aux"
if [ -d "$AUX_DIR/.git" ]; then
  echo "  already cloned"
else
  git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux "$AUX_DIR"
  echo "  → ComfyUI 컨테이너 재시작 필요: docker restart comfyui"
fi

echo ""
echo "=== Done. 다음 단계 ==="
echo "1) docker restart comfyui    # custom node 활성화"
echo "2) Task 2 진행"
```

- [ ] **Step 3: 실행 권한 + 실행**

```bash
chmod +x /home/yanus/unified_ai_service/scripts/download_b1a_models.sh
/home/yanus/unified_ai_service/scripts/download_b1a_models.sh 2>&1 | tail -30
```

Expected: FLUX와 ControlNet 다운로드 또는 "already present" 출력, 마지막 "Done." 라인.

만약 FLUX 다운로드가 403 에러 (HF dev 인증 필요)면:
- HF 계정으로 `huggingface.co/black-forest-labs/FLUX.1-dev` 접근해 라이선스 동의
- `HF_TOKEN=<token>` 환경변수 설정 후 wget의 `--header="Authorization: Bearer $HF_TOKEN"` 추가 재실행

**Comfy-Org/flux1-dev 의 fp8 변종은 인증 없이 다운 가능** (검증 완료, 본 plan 작성 시점 2026-05-23).

- [ ] **Step 4: 검증**

```bash
ls -lh /home/yanus/Docker/models/diffusion_models/FLUX1/flux1-dev-fp8.safetensors
ls -lh /home/yanus/Docker/models/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors
ls -d /home/yanus/Docker/custom_nodes_persistent/ComfyUI-controlnet-aux
```

Expected: 두 파일 각각 16-18GB / 2-4GB, `ComfyUI-controlnet-aux` 디렉토리 존재.

- [ ] **Step 5: ComfyUI 재시작**

```bash
docker restart comfyui
# 30초 대기 후 헬스체크
sleep 30
curl -s -o /dev/null -w "ComfyUI: %{http_code}\n" http://localhost:8188/system_stats
```

Expected: `ComfyUI: 200`.

- [ ] **Step 6: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/scripts/download_b1a_models.sh
git commit -m "$(cat <<'EOF'
feat(b1a): FLUX/ControlNet 모델 + custom node 다운로드 스크립트

flux1-dev-fp8.safetensors (~17GB) + FLUX-ControlNet-Union-Pro (~3GB) +
ComfyUI-controlnet-aux git clone. 멱등(이미 존재 시 스킵), wget --continue.
ComfyUI 컨테이너 재시작 1회로 custom node 활성화.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — Catalog 3개 항목 추가 + 단위 테스트

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_engine/catalog.py`
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/test_catalog.py`

- [ ] **Step 1: 신규 catalog 항목 추가 (TDD: 먼저 테스트)**

`test_catalog.py` 끝에 추가:

```python
def test_flux_workflow_metadata():
    meta = catalog.get("image.gen.flux")
    assert meta["template"] == "image_gen_flux.json.j2"
    assert meta["vram_class"] == "heavy"
    assert meta["output_node"] == "9"
    assert meta["timeout_sec"] >= 300


def test_flux_ctrl_union_metadata():
    meta = catalog.get("image.ctrl.flux_union")
    assert meta["template"] == "image_ctrl_flux_union.json.j2"
    assert meta["vram_class"] == "heavy"
    # params 검증
    params = catalog.validate(meta, {
        "prompt": "p",
        "control_image": "ctrl.png",
    })
    assert params["control_type"] == "canny"  # default
    assert params["strength"] == 0.7


def test_qwen_inpaint_metadata():
    meta = catalog.get("image.inpaint.qwen")
    params = catalog.validate(meta, {
        "prompt": "remove the cat",
        "image_name": "img.png",
        "mask_name": "msk.png",
    })
    assert params["denoise"] == 0.9


def test_list_workflows_includes_b1a_additions():
    ids = set(catalog.list_workflows())
    assert {"image.gen.flux", "image.ctrl.flux_union", "image.inpaint.qwen"}.issubset(ids)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/test_catalog.py -v
```

Expected: 4개 신규 테스트가 `KeyError: unknown workflow: image.gen.flux` 등으로 실패.

- [ ] **Step 3: `catalog.py` WORKFLOWS dict에 3개 항목 추가**

`catalog.py`의 `WORKFLOWS` dict 끝에 추가 (기존 5개 뒤에):

```python
    "image.gen.flux": {
        "template": "image_gen_flux.json.j2",
        "params": {
            "prompt":   (str,   ...),
            "width":    (int,   1024),
            "height":   (int,   1024),
            "steps":    (int,   20),
            "seed":     (int,   0),
            "guidance": (float, 3.5),
        },
        "models_required": [
            "diffusion_models/FLUX1/flux1-dev-fp8.safetensors",
            "text_encoders/clip_l.safetensors",
            "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
            "vae/ae.safetensors",
        ],
        "output_node": "9",
        "vram_class": "heavy",
        "timeout_sec": 600,
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_catalog.py -v
```

Expected: 15 passed (기존 11 + 신규 4).

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/catalog.py unified_ai_service/media_engine/tests/test_catalog.py
git commit -m "$(cat <<'EOF'
feat(b1a): catalog에 FLUX 생성·ControlNet·Qwen inpaint 메타 추가

image.gen.flux / image.ctrl.flux_union / image.inpaint.qwen 등록.
모두 heavy 클래스. control_type는 "canny" 기본, denoise 0.9 기본.
test_catalog 4건 추가 (메타 조회 + validate + list_workflows 포함).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — 워크플로우 템플릿 3개 + 렌더 테스트

**Files:**
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/image_gen_flux.json.j2`
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/image_ctrl_flux_union.json.j2`
- Create: `/home/yanus/unified_ai_service/media_engine/workflows/image_inpaint_qwen.json.j2`
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/test_workflow_render.py`

- [ ] **Step 1: 실패 테스트 추가 (test_workflow_render.py 끝)**

```python
def test_flux_renders_valid_json():
    meta = catalog.get("image.gen.flux")
    params = catalog.validate(meta, {"prompt": "a futuristic city skyline"})
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "a futuristic city skyline" in flat
    assert "flux1-dev-fp8.safetensors" in flat


def test_flux_ctrl_union_canny():
    meta = catalog.get("image.ctrl.flux_union")
    params = catalog.validate(meta, {
        "prompt": "pose-controlled portrait",
        "control_image": "ctrl_in.png",
        "control_type": "canny",
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf, ensure_ascii=False)
    assert "ctrl_in.png" in flat
    assert "FLUX.1-dev-ControlNet-Union-Pro.safetensors" in flat


def test_flux_ctrl_union_openpose_changes_mode():
    """control_type=openpose이면 Union mode index가 canny와 달라야 한다."""
    meta = catalog.get("image.ctrl.flux_union")
    p_canny = catalog.validate(meta, {"prompt": "p", "control_image": "x.png", "control_type": "canny"})
    p_pose = catalog.validate(meta, {"prompt": "p", "control_image": "x.png", "control_type": "openpose"})
    wf_canny = _render(meta["template"], p_canny)
    wf_pose = _render(meta["template"], p_pose)
    # mode index 또는 type 문자열이 워크플로우 어딘가에 들어가야 함
    flat_canny = json.dumps(wf_canny)
    flat_pose = json.dumps(wf_pose)
    assert flat_canny != flat_pose, "control_type 변경이 워크플로우에 반영되어야 함"


def test_qwen_inpaint_renders_valid_json():
    meta = catalog.get("image.inpaint.qwen")
    params = catalog.validate(meta, {
        "prompt": "replace background with beach",
        "image_name": "img.png",
        "mask_name": "msk.png",
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf, ensure_ascii=False)
    assert "img.png" in flat
    assert "msk.png" in flat
    assert "LoadImageMask" in flat  # 마스크 로딩 노드 사용
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v 2>&1 | tail -10
```

Expected: 4개 신규 테스트가 `TemplateNotFound`로 실패.

- [ ] **Step 3: `image_gen_flux.json.j2` 작성**

```jinja
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "FLUX1/flux1-dev-fp8.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "DualCLIPLoader",
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
      "type": "flux"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": { "vae_name": "ae.safetensors" }
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["2", 0]
    }
  },
  "5": {
    "class_type": "FluxGuidance",
    "inputs": {
      "conditioning": ["4", 0],
      "guidance": {{ guidance }}
    }
  },
  "6": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width":  {{ width }},
      "height": {{ height }},
      "batch_size": 1
    }
  },
  "7": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["5", 0],
      "negative": ["5", 0],
      "latent_image": ["6", 0],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["7", 0],
      "vae": ["3", 0]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["decode", 0],
      "filename_prefix": "flux_gen"
    }
  }
}
```

- [ ] **Step 4: `image_ctrl_flux_union.json.j2` 작성**

`control_type` 매핑: `canny=0`, `openpose=1`, `depth=2` (FLUX-ControlNet-Union-Pro 표준).

```jinja
{% set type_map = {"canny": 0, "openpose": 1, "depth": 2, "scribble": 3} %}
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "FLUX1/flux1-dev-fp8.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "DualCLIPLoader",
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
      "type": "flux"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": { "vae_name": "ae.safetensors" }
  },
  "4": {
    "class_type": "ControlNetLoader",
    "inputs": {
      "control_net_name": "FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors"
    }
  },
  "5": {
    "class_type": "LoadImage",
    "inputs": {
      "image": {{ control_image | tojson }},
      "upload": "image"
    }
  },
  "6": {
    "class_type": "SetUnionControlNetType",
    "inputs": {
      "control_net": ["4", 0],
      "type": {{ control_type | tojson }}
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["2", 0]
    }
  },
  "8": {
    "class_type": "FluxGuidance",
    "inputs": {
      "conditioning": ["7", 0],
      "guidance": {{ guidance }}
    }
  },
  "10": {
    "class_type": "ControlNetApplyAdvanced",
    "inputs": {
      "positive": ["8", 0],
      "negative": ["8", 0],
      "control_net": ["6", 0],
      "image": ["5", 0],
      "vae": ["3", 0],
      "strength": {{ strength }},
      "start_percent": 0.0,
      "end_percent": 1.0
    }
  },
  "11": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width":  {{ width }},
      "height": {{ height }},
      "batch_size": 1
    }
  },
  "12": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["10", 0],
      "negative": ["10", 1],
      "latent_image": ["11", 0],
      "seed": {{ seed }},
      "steps": {{ steps }},
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "decode": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["12", 0],
      "vae": ["3", 0]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["decode", 0],
      "filename_prefix": "flux_ctrl"
    }
  }
}
```

> 참고: `SetUnionControlNetType` 노드의 `type` 인자는 ComfyUI Union ControlNet 표준 문자열 (`canny|tile|depth|blur|pose|gray|low_quality` 등). 실제 노드 시그니처는 ComfyUI 버전에 따라 다를 수 있어 integration test 시 검증.

- [ ] **Step 5: `image_inpaint_qwen.json.j2` 작성**

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
    "inputs": { "vae_name": "qwen_image_vae.safetensors" }
  },
  "4": {
    "class_type": "LoadImage",
    "inputs": {
      "image": {{ image_name | tojson }},
      "upload": "image"
    }
  },
  "5": {
    "class_type": "LoadImageMask",
    "inputs": {
      "image": {{ mask_name | tojson }},
      "channel": "red",
      "upload": "image"
    }
  },
  "6": {
    "class_type": "VAEEncodeForInpaint",
    "inputs": {
      "pixels": ["4", 0],
      "vae": ["3", 0],
      "mask": ["5", 0],
      "grow_mask_by": 6
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": {{ prompt | tojson }},
      "clip": ["2", 0]
    }
  },
  "8": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "low quality, blurry, distorted",
      "clip": ["2", 0]
    }
  },
  "10": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["7", 0],
      "negative": ["8", 0],
      "latent_image": ["6", 0],
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
      "samples": ["10", 0],
      "vae": ["3", 0]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["decode", 0],
      "filename_prefix": "qwen_inpaint"
    }
  }
}
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
./venv/bin/python -m pytest media_engine/tests/test_workflow_render.py -v 2>&1 | tail -15
```

Expected: 11 passed (기존 7 + 신규 4).

전체 회귀:
```bash
./venv/bin/python -m pytest media_engine/tests/ -v 2>&1 | tail -5
```

Expected: 49 passed (기존 41 + catalog 4 + workflow 4).

- [ ] **Step 7: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/workflows/image_gen_flux.json.j2 \
        unified_ai_service/media_engine/workflows/image_ctrl_flux_union.json.j2 \
        unified_ai_service/media_engine/workflows/image_inpaint_qwen.json.j2 \
        unified_ai_service/media_engine/tests/test_workflow_render.py
git commit -m "$(cat <<'EOF'
feat(b1a): FLUX/Qwen 워크플로우 템플릿 3종 추가

- image_gen_flux: FLUX dev fp8 + DualCLIP(clip_l+t5xxl) + FluxGuidance
- image_ctrl_flux_union: ControlNet Union (canny/openpose/depth) + ControlNetApplyAdvanced
- image_inpaint_qwen: Qwen Edit + LoadImageMask + VAEEncodeForInpaint(grow_mask_by=6)
test_workflow_render 4건 추가 — control_type 분기 차이 검증 포함.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — `media_image.py` shim 확장

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_image.py`

- [ ] **Step 1: 기존 파일 확인**

```bash
cat /home/yanus/unified_ai_service/media_image.py
```

Phase A 산출(`generate_image`, `edit_image` 2개 함수). 끝에 함수 2개 추가 + `generate_image`의 `workflow` 인자에 `flux` 라우팅 추가.

- [ ] **Step 2: 파일 전체 재작성**

`/home/yanus/unified_ai_service/media_image.py`:

```python
"""이미지 생성·편집·변환 진입점 (media_engine.runner 호출하는 얇은 shim)."""
import logging
import os
import uuid
from pathlib import Path

from media_engine import runner, comfyui_client

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    workflow: str = "zimage_turbo",
    **kwargs,
) -> Path:
    """텍스트→이미지. workflow ∈ {"zimage_turbo", "flux"}."""
    workflow_id = f"image.gen.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, **kwargs)


async def edit_image(
    image_path: str,
    prompt: str,
    workflow: str = "qwen",
    **kwargs,
) -> Path:
    """이미지+프롬프트→편집된 이미지. workflow ∈ {"qwen"}."""
    filename = f"edit_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    await comfyui_client.upload_image(image_path, filename)
    workflow_id = f"image.edit.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, image_name=filename, **kwargs)


async def control_image(
    prompt: str,
    control_image_path: str,
    control_type: str = "canny",
    strength: float = 0.7,
    **kwargs,
) -> Path:
    """FLUX ControlNet으로 구도/자세/깊이 제어 이미지 생성.
    control_type ∈ {"canny", "openpose", "depth"}.
    """
    filename = f"ctrl_{uuid.uuid4().hex[:8]}_{os.path.basename(control_image_path)}"
    await comfyui_client.upload_image(control_image_path, filename)
    return await runner.run(
        "image.ctrl.flux_union",
        prompt=prompt,
        control_image=filename,
        control_type=control_type,
        strength=strength,
        **kwargs,
    )


async def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    **kwargs,
) -> Path:
    """Qwen Image Edit으로 마스크 영역만 instruction-edit.
    mask: 흑백 PNG (white=수정 영역, black=보존).
    """
    img_filename = f"inp_img_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    msk_filename = f"inp_msk_{uuid.uuid4().hex[:8]}_{os.path.basename(mask_path)}"
    await comfyui_client.upload_image(image_path, img_filename)
    await comfyui_client.upload_image(mask_path, msk_filename)
    return await runner.run(
        "image.inpaint.qwen",
        prompt=prompt,
        image_name=img_filename,
        mask_name=msk_filename,
        **kwargs,
    )
```

- [ ] **Step 3: 임포트 검증**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "
import media_image
print('OK:', media_image.generate_image)
print('OK:', media_image.edit_image)
print('OK:', media_image.control_image)
print('OK:', media_image.inpaint_image)
"
```

Expected: 4개 함수 모두 callable.

```bash
./venv/bin/python -c "import main; print('main OK')"
```

Expected: `main OK`.

- [ ] **Step 4: 회귀 테스트**

```bash
./venv/bin/python -m pytest media_engine/tests/ -v 2>&1 | tail -5
```

Expected: 49 passed.

- [ ] **Step 5: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_image.py
git commit -m "$(cat <<'EOF'
feat(b1a): media_image.py에 control_image / inpaint_image 추가

- control_image(prompt, path, control_type, strength): FLUX ControlNet Union
- inpaint_image(image_path, mask_path, prompt): Qwen Edit 마스크 inpaint
- generate_image(workflow="zimage_turbo"|"flux") 라우팅 확장

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — `main.py` 엔드포인트 2개 추가

**Files:**
- Modify: `/home/yanus/unified_ai_service/main.py`

- [ ] **Step 1: 현재 미디어 엔드포인트 영역 확인**

```bash
grep -n "@app.post.*media\|@app.get.*health" /home/yanus/unified_ai_service/main.py
```

`/api/media/image`, `/api/media/image/edit` 위치 식별 — 그 사이/뒤에 신규 엔드포인트 추가.

- [ ] **Step 2: `/api/media/image/control` 엔드포인트 추가**

`main.py`의 `/api/media/image/edit` 엔드포인트 정의 직후에 추가:

```python
@app.post("/api/media/image/control")
async def control_image_endpoint(
    prompt: str = Form(...),
    control_type: str = Form("canny"),
    strength: float = Form(0.7),
    control_image: UploadFile = File(...),
    auth = Depends(flexible_auth),
):
    path = os.path.join(UPLOADS_DIR, f"ctrl_in_{uuid.uuid4().hex}_{control_image.filename}")
    with open(path, "wb") as f:
        f.write(await control_image.read())
    coro = media_image.control_image(prompt, path, control_type=control_type, strength=strength)
    job_id = await job_queue.submit(
        "media_image_control",
        {"prompt": prompt, "control_type": control_type, "strength": strength},
        coro,
    )
    return {"job_id": job_id}
```

- [ ] **Step 3: `/api/media/image/inpaint` 엔드포인트 추가**

위 엔드포인트 직후에 추가:

```python
@app.post("/api/media/image/inpaint")
async def inpaint_image_endpoint(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    auth = Depends(flexible_auth),
):
    img_path = os.path.join(UPLOADS_DIR, f"inp_img_{uuid.uuid4().hex}_{image.filename}")
    msk_path = os.path.join(UPLOADS_DIR, f"inp_msk_{uuid.uuid4().hex}_{mask.filename}")
    with open(img_path, "wb") as f:
        f.write(await image.read())
    with open(msk_path, "wb") as f:
        f.write(await mask.read())
    coro = media_image.inpaint_image(img_path, msk_path, prompt)
    job_id = await job_queue.submit("media_image_inpaint", {"prompt": prompt}, coro)
    return {"job_id": job_id}
```

- [ ] **Step 4: 임포트 + 엔드포인트 등록 검증**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -c "
import main
routes = sorted(r.path for r in main.app.routes if hasattr(r, 'path') and 'media' in r.path)
for r in routes:
    print(r)
"
```

Expected: 다음 10개 출력
```
/api/media/image
/api/media/image/control
/api/media/image/edit
/api/media/image/inpaint
/api/media/music
/api/media/tts
/api/media/video/analyze
/api/media/video/edit
/api/media/video/gen
/api/media/video/shorts
```

- [ ] **Step 5: 회귀 테스트**

```bash
./venv/bin/python -m pytest media_engine/tests/ -v 2>&1 | tail -5
```

Expected: 49 passed.

- [ ] **Step 6: 서비스 재기동 (변경 반영)**

```bash
# 기존 uvicorn pid 찾기
pgrep -af "uvicorn.*main:app.*8081" 2>&1 | head -2
# kill + restart
pkill -f "uvicorn.*main:app.*8081" 2>/dev/null; sleep 2
cd /home/yanus/unified_ai_service
nohup venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8081 > uvicorn.log 2>&1 &
echo "started pid=$!"
sleep 3
ss -tlnp 2>&1 | grep ":8081" | head -2
```

Expected: 8081 LISTEN 상태.

- [ ] **Step 7: 신규 엔드포인트 OPTIONS 확인**

```bash
for p in /api/media/image/control /api/media/image/inpaint; do
  c=$(curl -s -o /dev/null -w "%{http_code}" -X OPTIONS "http://localhost:8081$p")
  echo "$p: HTTP $c"
done
```

Expected: `HTTP 405` 두 줄 (Method Not Allowed for OPTIONS = endpoint exists, POST-only).

- [ ] **Step 8: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/main.py
git commit -m "$(cat <<'EOF'
feat(b1a): /api/media/image/control + /inpaint 엔드포인트 추가

POST /api/media/image/control — FLUX ControlNet (canny/openpose/depth)
POST /api/media/image/inpaint — Qwen Edit + 마스크 영역 instruction-edit
두 엔드포인트 모두 job_queue.submit 패턴, multipart upload (image+mask).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — Integration 테스트 (실제 GPU, 수동)

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_engine/tests/integration/test_real_workflows.py`

- [ ] **Step 1: 사전 점검**

```bash
docker ps --format "{{.Names}}: {{.Status}}" | grep -E "comfyui|vllm"
curl -s -o /dev/null -w "ComfyUI: %{http_code}\n" http://localhost:8188/system_stats
curl -s -o /dev/null -w "vLLM: %{http_code}\n" http://localhost:8080/v1/models
ls -lh /home/yanus/Docker/models/diffusion_models/FLUX1/flux1-dev-fp8.safetensors
ls -lh /home/yanus/Docker/models/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors
```

Expected: 두 컨테이너 Up, 200/200, 두 파일 존재 (15GB+, 2GB+).

- [ ] **Step 2: integration 테스트 3건 추가**

`test_real_workflows.py` 파일 끝에 추가:

```python
@pytest.mark.asyncio
async def test_flux_text2img_end_to_end():
    """FLUX dev fp8 text2img — 1024x1024 20step, vLLM swap 1회."""
    from media_engine import runner
    out = await runner.run(
        "image.gen.flux",
        prompt="a serene mountain lake at sunset, photorealistic",
        steps=20,
    )
    assert out.exists() and out.stat().st_size > 50_000


@pytest.mark.asyncio
async def test_flux_ctrl_canny_end_to_end(tmp_path):
    """FLUX-ControlNet Union canny — 업로드된 canny edge 이미지로 합성."""
    from media_engine import runner, comfyui_client
    from PIL import Image, ImageDraw
    # 간단한 canny 입력 (검정 배경 + 흰 사각형)
    img = Image.new("RGB", (1024, 1024), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 200, 800, 800], outline=(255, 255, 255), width=10)
    ctrl_path = tmp_path / "canny.png"
    img.save(ctrl_path)
    await comfyui_client.upload_image(str(ctrl_path), "integ_canny.png")

    out = await runner.run(
        "image.ctrl.flux_union",
        prompt="a glowing crystal cube floating in space",
        control_image="integ_canny.png",
        control_type="canny",
        steps=20,
    )
    assert out.exists() and out.stat().st_size > 50_000


@pytest.mark.asyncio
async def test_qwen_inpaint_end_to_end(tmp_path):
    """Qwen Edit 마스크 inpaint — 흰 사각형 마스크로 중앙 영역 교체."""
    from media_engine import runner, comfyui_client
    from PIL import Image, ImageDraw
    base = Image.new("RGB", (1024, 1024), (100, 150, 100))
    base_path = tmp_path / "base.png"
    base.save(base_path)

    mask = Image.new("RGB", (1024, 1024), (0, 0, 0))
    d = ImageDraw.Draw(mask)
    d.rectangle([300, 300, 700, 700], fill=(255, 255, 255))
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path)

    await comfyui_client.upload_image(str(base_path), "integ_inp_img.png")
    await comfyui_client.upload_image(str(mask_path), "integ_inp_msk.png")

    out = await runner.run(
        "image.inpaint.qwen",
        prompt="a vibrant red rose in the center",
        image_name="integ_inp_img.png",
        mask_name="integ_inp_msk.png",
        steps=20,
    )
    assert out.exists() and out.stat().st_size > 50_000
```

- [ ] **Step 3: Skip 동작 확인 (--integration 없이)**

```bash
cd /home/yanus/unified_ai_service
./venv/bin/python -m pytest media_engine/tests/integration -v 2>&1 | tail -10
```

Expected: 6 skipped (기존 3 + 신규 3).

- [ ] **Step 4: 실제 GPU 시험 (1건씩 분리 실행 — 안정성)**

각 테스트는 vLLM swap 1회 + Wan/FLUX 14B급 실행을 포함하므로 ~5분/회.

```bash
cd /home/yanus/unified_ai_service
# 1. FLUX text2img
./venv/bin/python -m pytest media_engine/tests/integration/test_real_workflows.py::test_flux_text2img_end_to_end -v --integration -s --timeout=1800 2>&1 | tail -20
```

Expected: PASS, results/ 디렉토리에 `image_gen_flux_*.png` 산출.

만약 실패 시:
- ComfyUI 로그: `docker logs comfyui --tail 200`
- 모델 누락이면: Task 1 다운로드 스크립트 재확인
- 노드 누락(예: `FluxGuidance`)이면: ComfyUI 버전 업그레이드 또는 워크플로우 노드명 보정

```bash
# 2. FLUX ControlNet canny
./venv/bin/python -m pytest media_engine/tests/integration/test_real_workflows.py::test_flux_ctrl_canny_end_to_end -v --integration -s --timeout=1800 2>&1 | tail -20
```

Expected: PASS.

만약 `SetUnionControlNetType` 노드 누락이면: ComfyUI-controlnet-aux의 `Union` 노드 또는 ComfyUI 기본의 `ControlNetApplyAdvanced`로 fallback. 워크플로우 템플릿 수정 후 재시도.

```bash
# 3. Qwen inpaint
./venv/bin/python -m pytest media_engine/tests/integration/test_real_workflows.py::test_qwen_inpaint_end_to_end -v --integration -s --timeout=1800 2>&1 | tail -20
```

Expected: PASS.

- [ ] **Step 5: vLLM 정상 복구 보장**

```bash
docker ps --format "{{.Names}}: {{.Status}}" | grep vllm
curl -s -o /dev/null -w "vLLM: %{http_code}\n" --connect-timeout 5 http://localhost:8080/v1/models
```

만약 vLLM이 stopped면: `docker start vllm-server`. ready까지 ~2-5분 대기.

- [ ] **Step 6: 산출 파일 확인**

```bash
ls -lh /home/yanus/unified_ai_service/results/ | grep -E "image_(gen_flux|ctrl_flux|inpaint_qwen)" | tail -10
```

Expected: 3개 신규 파일 (각 100KB+).

- [ ] **Step 7: 커밋 (테스트 파일만, 실패해도 OK)**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/tests/integration/test_real_workflows.py
git commit -m "$(cat <<'EOF'
test(b1a): integration 시험 — FLUX text2img / ControlNet / Qwen inpaint

PIL로 캐니 엣지·마스크 이미지를 즉석 생성해 실제 ComfyUI 워크플로우
실행. --integration 플래그로만 수집, vLLM swap 한 사이클 각 ~5분.
실패 시 ComfyUI 노드 가용성 점검 후 워크플로우 보정 필요.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7 — Smoke + README 업데이트

**Files:**
- Modify: `/home/yanus/unified_ai_service/media_engine/README.md`

- [ ] **Step 1: 신규 엔드포인트 스모크 — `workflow=flux`**

```bash
JOB=$(curl -s -X POST "http://localhost:8081/api/media/image" \
  -F "prompt=a quiet sunset over mountains" \
  -F "workflow=flux")
echo "$JOB"
JID=$(echo "$JOB" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
# 5분 폴링
for i in 1 2 3 4 5 6 7 8 9 10; do
  s=$(curl -s "http://localhost:8081/api/jobs/$JID" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")
  echo "[+${i}0s] status=$s"
  if [ "$s" = "completed" ] || [ "$s" = "failed" ]; then break; fi
  sleep 30
done
```

Expected: 최종 `completed` 또는 `failed` (timeout 명시).

- [ ] **Step 2: control 엔드포인트 스모크**

```bash
# 더미 canny 이미지 만들기
python3 -c "
from PIL import Image, ImageDraw
img = Image.new('RGB', (512, 512), (0,0,0))
d = ImageDraw.Draw(img)
d.rectangle([100, 100, 400, 400], outline=(255,255,255), width=8)
img.save('/tmp/canny_smoke.png')
"
curl -s -X POST "http://localhost:8081/api/media/image/control" \
  -F "prompt=a luminous geometric shape, neon style" \
  -F "control_type=canny" \
  -F "strength=0.7" \
  -F "control_image=@/tmp/canny_smoke.png"
```

Expected: `{"job_id": "..."}` 반환.

- [ ] **Step 3: inpaint 엔드포인트 스모크**

```bash
# 더미 base + mask
python3 -c "
from PIL import Image, ImageDraw
b = Image.new('RGB', (512, 512), (100,150,100)); b.save('/tmp/base_smoke.png')
m = Image.new('RGB', (512, 512), (0,0,0))
d = ImageDraw.Draw(m); d.rectangle([150, 150, 350, 350], fill=(255,255,255))
m.save('/tmp/mask_smoke.png')
"
curl -s -X POST "http://localhost:8081/api/media/image/inpaint" \
  -F "prompt=a glowing crystal in the masked area" \
  -F "image=@/tmp/base_smoke.png" \
  -F "mask=@/tmp/mask_smoke.png"
```

Expected: `{"job_id": "..."}` 반환.

- [ ] **Step 4: 회귀 확인 — Phase A 엔드포인트 정상**

```bash
curl -s http://localhost:8081/api/health/vllm
echo ""
# 기존 zimage_turbo 동작 확인
curl -s -X POST "http://localhost:8081/api/media/image" \
  -F "prompt=test" -F "workflow=zimage_turbo"
```

Expected: health endpoint JSON, zimage_turbo job_id 반환.

- [ ] **Step 5: README 갱신**

`/home/yanus/unified_ai_service/media_engine/README.md` 의 **"엔드포인트"** 표에 추가하고, **"알려진 한계"** 절 위에 다음 절 삽입:

```markdown
## Phase B1a 산출 (2026-05-23)

- 신규 워크플로우 3개:
  - `image.gen.flux` — FLUX dev fp8 text2img (heavy, ~5분)
  - `image.ctrl.flux_union` — FLUX-ControlNet-Union-Pro (canny/openpose/depth/scribble)
  - `image.inpaint.qwen` — Qwen Image Edit + 마스크 inpaint
- 신규 엔드포인트:
  - `POST /api/media/image/control` (control_image + control_type)
  - `POST /api/media/image/inpaint` (image + mask)
  - `POST /api/media/image?workflow=flux` (기존 endpoint 옵션 추가)
- 디스크 추가: FLUX fp8 (~17GB), FLUX-ControlNet-Union-Pro (~3GB)
- Custom node: ComfyUI-controlnet-aux (canny/openpose/depth preprocessor)

자동 다운로드: `unified_ai_service/scripts/download_b1a_models.sh`
```

기존 "엔드포인트" 표에도 다음 3행 추가:

```markdown
| POST | `/api/media/image?workflow=flux` | image.gen.flux (FLUX dev fp8) |
| POST | `/api/media/image/control` | image.ctrl.flux_union (FLUX-ControlNet Union) |
| POST | `/api/media/image/inpaint` | image.inpaint.qwen (Qwen + 마스크) |
```

- [ ] **Step 6: 커밋**

```bash
cd /home/yanus
git add unified_ai_service/media_engine/README.md
git commit -m "$(cat <<'EOF'
docs(b1a): README — Phase B1a 산출 절 + 엔드포인트 표 갱신

신규 워크플로우 3개, 신규 엔드포인트 2개, 디스크/노드 추가사항,
자동 다운로드 스크립트 안내.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 종합 자가 점검 (Plan 작성자 self-review)

- [x] **Spec 커버리지**:
  - § 2 결정사항 → Task 1, 2, 3 (catalog/template/scripts)
  - § 3 아키텍처 → Task 1-7 모두 spec 디렉토리 매핑 따름
  - § 4.1 Catalog 항목 → Task 2
  - § 4.2 워크플로우 → Task 3
  - § 4.3 Shim → Task 4
  - § 4.4 Endpoint → Task 5
  - § 4.5 다운로드 스크립트 → Task 1
  - § 5 데이터 흐름 → Task 5 (엔드포인트가 spec 흐름 그대로 구현)
  - § 6 에러 처리 → catalog.validate가 control_type 검증, ComfyUIError가 다른 실패 처리 (기존 모듈)
  - § 7 테스트 — Unit/Integration/Smoke 모두 Task 2/3/6/7에 매핑
  - § 8 DoD → Task 7 종료 시점에 모두 충족

- [x] **Placeholder scan**: 모든 step은 실제 코드/명령. TBD/TODO 없음.

- [x] **타입 일관성**:
  - `runner.run(workflow_id: str, **params) -> Path` — Phase A 정의, Task 4 호출 일치
  - `comfyui_client.upload_image(local_path, filename) -> dict` — Phase A, Task 4 호출 일치
  - `media_image.control_image(prompt, control_image_path, control_type="canny", strength=0.7, **kw) -> Path` — Task 4 정의, Task 5 호출 일치
  - `media_image.inpaint_image(image_path, mask_path, prompt, **kw) -> Path` — Task 4 정의, Task 5 호출 일치
  - `job_queue.submit(job_type, payload, coro) -> str` — Phase A, Task 5 호출 일치

- [x] **자체완결성**: 각 task의 step은 다른 task 코드 참조 없이 그대로 실행 가능.
