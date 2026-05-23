# Phase B1a — 이미지 변환 강화 (FLUX ControlNet + Qwen Inpaint) 설계

**작성일**: 2026-05-23
**범위**: B1a only. B1b(생성 강화)와 B1c(분석 강화)는 별도 spec.
**전제**: Phase A 산출(`media_engine/` 패키지, catalog/runner/comfyui_client/gpu_arbiter/window/job_queue) 활용.

## 1. 목표

`unified_ai_service`의 이미지 변환 능력을 상용 수준으로 강화한다.

1. **FLUX 기반 ControlNet** (canny / openpose / depth) — 구도·자세·깊이로 이미지 합성을 제어
2. **Qwen Image Edit + 마스크** — 사용자가 흑백 마스크를 업로드해 특정 영역만 instruction-edit
3. **FLUX text2img** — Phase A의 zimage_turbo보다 한 단계 고품질 텍스트→이미지

## 2. 결정사항 (브레인스토밍 산출)

| 결정 | 값 | 근거 |
|---|---|---|
| 베이스 모델 | FLUX dev fp8 + Qwen Image Edit 2509 | 디스크 잔여 132GB → ~17GB 추가 가능 |
| ControlNet | FLUX.1-dev-ControlNet-Union-Pro (~3GB) | union 모델로 canny/pose/depth 단일 weight |
| FLUX 로딩 | fp8 safetensors (GGUF 아님) | ComfyUI 기본 UNETLoader, 커스텀 노드 불필요 |
| Preprocessor | ComfyUI-controlnet-aux 커스텀 노드 | canny/openpose/depth 전처리 통합 |
| 워크플로우 형식 | Phase A 일관: ComfyUI API JSON + Jinja | catalog 단일 출처 |
| 잡 분류 | 모두 heavy (vLLM swap 필요) | FLUX/Qwen Edit 14B급 fp8 |
| 마스크 형식 | 흑백 PNG (white=수정 영역, black=보존) | ComfyUI LoadImageMask 표준 |

## 3. 아키텍처

기존 `media_engine/` 위에 누적. 신규 파일만 명시:

```
unified_ai_service/
├── media_engine/
│   ├── workflows/
│   │   ├── image_gen_flux.json.j2          # 신규
│   │   ├── image_ctrl_flux_union.json.j2   # 신규 — canny/pose/depth 분기
│   │   ├── image_inpaint_qwen.json.j2      # 신규
│   │   └── (기존 5개 유지)
│   ├── catalog.py                           # 항목 3개 추가
│   └── tests/test_workflow_render.py        # 테스트 3건 추가
├── media_image.py                           # control_image, inpaint_image 함수 추가
├── main.py                                  # /api/media/image/control, /inpaint 엔드포인트 추가
└── scripts/
    └── download_b1a_models.sh               # 신규 — 다운로드 자동화
```

ComfyUI 측 변경:
- `Docker/models/diffusion_models/FLUX1/flux1-dev-fp8.safetensors` 추가
- `Docker/models/controlnet/FLUX/Shakker-Labs-FLUX.1-dev-ControlNet-Union-Pro.safetensors` 추가
- `Docker/custom_nodes_persistent/ComfyUI-controlnet-aux/` (git clone)

## 4. 컴포넌트 상세

### 4.1 Catalog 항목 (신규 3개)

```python
WORKFLOWS = {
    # ... (Phase A 항목 5개 유지)
    "image.gen.flux": {
        "template": "image_gen_flux.json.j2",
        "params": {
            "prompt":  (str, ...),
            "width":   (int, 1024),
            "height":  (int, 1024),
            "steps":   (int, 20),
            "seed":    (int, 0),
            "guidance":(float, 3.5),
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
            "prompt":         (str, ...),
            "control_image":  (str, ...),    # ComfyUI 업로드된 파일명
            "control_type":   (str, "canny"),# canny|openpose|depth
            "strength":       (float, 0.7),
            "width":          (int, 1024),
            "height":         (int, 1024),
            "steps":          (int, 20),
            "seed":           (int, 0),
            "guidance":       (float, 3.5),
        },
        "models_required": [
            "diffusion_models/FLUX1/flux1-dev-fp8.safetensors",
            "controlnet/FLUX/Shakker-Labs-FLUX.1-dev-ControlNet-Union-Pro.safetensors",
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
            "prompt":      (str, ...),
            "image_name":  (str, ...),
            "mask_name":   (str, ...),
            "steps":       (int, 20),
            "seed":        (int, 0),
            "denoise":     (float, 0.9),
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
}
```

### 4.2 워크플로우 템플릿 핵심

- `image_gen_flux.json.j2`: UNETLoader(FLUX) → DualCLIPLoader(clip_l + t5xxl) → CLIPTextEncode → EmptyLatentImage → KSampler(euler/simple, guidance via FluxGuidance) → VAEDecode → SaveImage
- `image_ctrl_flux_union.json.j2`: 위에 추가로 LoadImage(control) → ControlNetLoader(Union) → ControlNetApplyAdvanced(strength, control_type을 setting으로). `control_type`은 Union 모델의 mode index로 매핑
- `image_inpaint_qwen.json.j2`: LoadImage(이미지) + LoadImageMask(마스크) → VAEEncodeForInpaint → KSampler → VAEDecode → SaveImage

### 4.3 Shim (`media_image.py`) 확장

```python
async def control_image(
    prompt: str,
    control_image_path: str,
    control_type: str = "canny",
    **kwargs,
) -> Path:
    """FLUX ControlNet으로 구도/자세/깊이 제어 이미지 생성."""
    filename = f"ctrl_{uuid.uuid4().hex[:8]}_{os.path.basename(control_image_path)}"
    await comfyui_client.upload_image(control_image_path, filename)
    return await runner.run(
        "image.ctrl.flux_union",
        prompt=prompt, control_image=filename, control_type=control_type, **kwargs,
    )

async def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    **kwargs,
) -> Path:
    """Qwen Image Edit으로 마스크 영역만 instruction-edit."""
    img_filename = f"inp_img_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    msk_filename = f"inp_msk_{uuid.uuid4().hex[:8]}_{os.path.basename(mask_path)}"
    await comfyui_client.upload_image(image_path, img_filename)
    await comfyui_client.upload_image(mask_path, msk_filename)
    return await runner.run(
        "image.inpaint.qwen",
        prompt=prompt, image_name=img_filename, mask_name=msk_filename, **kwargs,
    )
```

기존 `generate_image(prompt, workflow="zimage_turbo")` 의 `workflow` 인자에 `"flux"` 추가.

### 4.4 Endpoint 추가 (`main.py`)

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
    job_id = await job_queue.submit("media_image_control", {"prompt": prompt, "control_type": control_type}, coro)
    return {"job_id": job_id}

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

### 4.5 모델 다운로드 스크립트

`unified_ai_service/scripts/download_b1a_models.sh`:

```bash
#!/bin/bash
set -e
MODELS_ROOT=/home/yanus/Docker/models
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "$MODELS_ROOT/diffusion_models/FLUX1"
mkdir -p "$MODELS_ROOT/controlnet/FLUX"
mkdir -p /home/yanus/Docker/custom_nodes_persistent

echo "[1/3] FLUX dev fp8..."
if [ ! -f "$MODELS_ROOT/diffusion_models/FLUX1/flux1-dev-fp8.safetensors" ]; then
  cd "$MODELS_ROOT/diffusion_models/FLUX1" && \
  wget --header="Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors
fi

echo "[2/3] FLUX-ControlNet Union Pro..."
if [ ! -f "$MODELS_ROOT/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors" ]; then
  cd "$MODELS_ROOT/controlnet/FLUX" && \
  wget https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors \
    -O FLUX.1-dev-ControlNet-Union-Pro.safetensors
fi

echo "[3/3] ComfyUI-controlnet-aux 노드..."
NODES=/home/yanus/Docker/custom_nodes_persistent
if [ ! -d "$NODES/ComfyUI-controlnet-aux" ]; then
  cd "$NODES" && git clone https://github.com/Fannovel16/comfyui_controlnet_aux ComfyUI-controlnet-aux
  echo "  → ComfyUI 컨테이너 재시작 필요: docker restart comfyui"
fi

echo "Done."
```

## 5. 데이터 흐름 (control 예)

```
client POST /api/media/image/control (prompt, control_type, control_image)
  ↓ main.py
파일 저장 → uploads/ctrl_in_*.png
  ↓ media_image.control_image(prompt, path, control_type)
comfyui_client.upload_image → ComfyUI input/
  ↓ runner.run("image.ctrl.flux_union", prompt=..., control_image=filename, control_type=...)
gpu_arbiter.acquire("heavy"):
  docker stop vllm-server
  Jinja render → ComfyUI POST /prompt → poll /history → fetch
  docker start vllm-server (lock 내, inline await)
results/image_ctrl_flux_union_*.png
  ↓ job_queue → update_job("completed", result="/api/results/...")
client GET /api/jobs/{id}
```

## 6. 에러 처리

| 실패 | 대응 |
|---|---|
| FLUX 모델 누락 | `catalog.check_models_present` 호출 시 사전 거부, 다운로드 안내 |
| `control_type` 유효값 외 | `catalog.validate` 단계에서 ValueError ("canny\|openpose\|depth") |
| 마스크 dimension ≠ 이미지 | ComfyUI 실패 → ComfyUIError 사용자 친화 메시지로 |
| FLUX OOM | vLLM swap이 작동했다면 가능성 낮음. 발생 시 `width/height` 축소 안내 |
| ComfyUI 커스텀 노드(controlnet_aux) 누락 | 워크플로우 submit 시 node_error 반환 → ComfyUIError, README의 설치 단계 안내 |

## 7. 테스트

### Unit (CI 가능)
- `test_workflow_render.py`에 3건 추가:
  - `test_flux_renders_valid_json`
  - `test_flux_ctrl_union_canny_renders` + `test_flux_ctrl_union_pose_renders`
  - `test_qwen_inpaint_renders_valid_json`
- `test_catalog.py`: 신규 3개 워크플로우의 `list_workflows` 검증

### Integration (수동, --integration)
- `test_flux_text2img_end_to_end` — FLUX text2img 1024² 20 step
- `test_flux_ctrl_canny_end_to_end` — canny ControlNet 1건
- `test_qwen_inpaint_end_to_end` — 마스크 inpaint 1건 (테스트 마스크 흰사각형)

### Smoke (배포 직전)
- `POST /api/media/image?workflow=flux` 200
- `POST /api/media/image/control` 200 + job 추적
- `POST /api/media/image/inpaint` 200 + job 추적
- 기존 9개 미디어 엔드포인트 회귀 없음

## 8. 완료 정의 (DoD)

1. **다운로드**: FLUX fp8 + FLUX-ControlNet-Union-Pro + controlnet_aux 모두 디스크에 존재
2. **워크플로우 3개** ComfyUI API 형식 + Jinja 렌더 → 모두 단위 테스트 통과
3. **Catalog 3개 항목** + validate 동작
4. **신규 2 엔드포인트** + `workflow=flux` 옵션 — 모두 HTTP 200 + job_id
5. **회귀** Phase A 9 엔드포인트 정상 동작
6. **README**(`media_engine/README.md`) 갱신: B1a 산출 추가

## 9. 범위 외 (Phase B1b/B1c로 이연)

- FLUX Lightning LoRA (4-step 가속) → B1b
- Janus-Pro 멀티모달 분석 → B1c
- Outpaint (캔버스 확장) → 별도 sub-project 또는 B1a 확장
- IP-Adapter (이미지 참조 합성) → 별도 sub-project
- Multi-ControlNet 결합 (canny + pose 동시) → B1a 확장

## 10. 위험

| 위험 | 완화 |
|---|---|
| FLUX 다운로드(17GB) 실패 | 스크립트가 멱등 (이미 존재 시 스킵), wget -c 재시도 |
| HF_TOKEN 누락 (FLUX dev는 인증 필요) | 스크립트가 환경변수 체크, 부재 시 명확한 안내 |
| controlnet_aux 노드 의존성 충돌 | 격리된 `custom_nodes_persistent/` 디렉토리 사용, 컨테이너 재시작 1회만 |
| ComfyUI cold start로 timeout 600s 도 부족 | Phase A에서 본 issue. catalog timeout_sec 600s + 첫 실행 시 warmup 한 번 권장 (README) |
| 디스크 잔여 132GB - 17GB - 3GB = 112GB | 충분 |
