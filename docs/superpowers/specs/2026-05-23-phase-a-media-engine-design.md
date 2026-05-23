# Phase A — `media_engine` 기반 인프라 설계

**작성일**: 2026-05-23
**대상**: `unified_ai_service`의 멀티모달 미디어 능력을 떠받칠 공용 엔진 신설
**범위**: Phase A only — Phase B(이미지/오디오/음성/비디오 능력 강화)와 Phase C(멀티모달 통합 UI)는 별도 spec

---

## 1. 배경과 목적

현재 `unified_ai_service`에는 이미지/오디오/비디오 생성 기능이 각자 다른 패턴으로 구현돼 있다.

- `media_image.py`: Python dict로 ComfyUI 워크플로우를 인라인으로 박아두고 ComfyUI를 호출
- `media_audio.py`: MusicGen은 transformers로 직접 로드, TTS는 `f5-tts_infer-cli` 서브프로세스
- `media_video.py`: 비디오 생성이 **mock** 상태 (`# MOCK implementation for safety in this sandbox`), 색깔 블록만 출력
- `lecture_service/orchestrator.py`: Wan2.2 워크플로우를 별도 함수로 가지고 별도 잡 상태 보관 (jobs.json)

이로 인해 (a) 워크플로우 수정 시마다 코드 수정 필요, (b) GPU 자원 경합으로 비디오 생성이 mock 처리됨, (c) 무빙윈도우 청크링 로직이 lecture_service에만 있어 다른 미디어로 일반화 불가, (d) 두 서비스의 잡 상태가 분리되어 사용자 체감 일관성 저하 같은 문제가 누적된 상태다.

Phase A는 후속 Phase B/C 의 모든 능력이 의존할 공용 기반을 마련한다.

## 2. 결정된 정책 (브레인스토밍 산출)

| 결정 | 값 | 근거 |
|---|---|---|
| 범위 분해 | Sub-project 분해 (A → B → C) | 10개 능력 + 통합 UI를 한 번에 spec화하면 품질 저하 위험 |
| GPU 정책 | 동적 vLLM 손님전환 | GB10 단일 GPU(128GB), Wan 14B + vLLM 동시 적재 불가 |
| 워크플로우 카탈로그 | ComfyUI native API JSON + Jinja 파라미터 | 디자이너가 ComfyUI GUI에서 직접 편집 가능, 코드 재배포 불필요 |
| 무빙윈도우 | 고정 청크 + 청크간 오버랩 | 품질·속도 균형. 비디오 5±1초, 음악 30±5초, 음성 문장단위 + 150ms 패드 |
| vLLM swap 방식 | `docker stop` → 작업 → `docker start` | 메모리 회수 확실, 구현 단순. 복구 ~30초 |
| 구조 접근 | 신규 `media_engine/` 패키지 + 어댑터 | 이후 모든 sub-project가 공유, 테스트 격리 |

## 3. 아키텍처

```
unified_ai_service/
├── media_engine/                       # 신규 패키지
│   ├── __init__.py
│   ├── workflows/                      # ComfyUI API JSON + Jinja 템플릿
│   │   ├── image_gen_flux.json.j2
│   │   ├── image_gen_zimage_turbo.json.j2
│   │   ├── image_edit_qwen.json.j2     # img2img / inpaint 기반
│   │   ├── video_i2v_wan22.json.j2     # Wan2.2 i2v 14B, 5초 청크
│   │   ├── video_s2v_wan22.json.j2     # Wan2.2 s2v 14B (lip-sync)
│   │   └── README.md                   # 추가 워크플로우 작성 가이드
│   ├── catalog.py                      # 워크플로우 메타데이터 단일 출처
│   ├── runner.py                       # render → submit → poll → fetch
│   ├── gpu_arbiter.py                  # asyncio.Lock + vLLM container 제어
│   ├── window.py                       # 청크링/last_frame/concat/crossfade
│   ├── job_queue.py                    # job_manager 위 직렬화 래퍼
│   ├── comfyui_client.py               # ComfyUI HTTP 클라이언트
│   └── tests/
│       ├── test_catalog.py
│       ├── test_runner_mock.py
│       ├── test_gpu_arbiter.py
│       ├── test_window.py
│       └── integration/
│           ├── test_image_gen.py       # 실제 ComfyUI 호출, GPU 필요
│           └── test_vllm_swap.py
├── media_image.py                      # media_engine을 호출하는 얇은 shim
├── media_audio.py                      # shim (MusicGen/F5-TTS 직호출 유지)
├── media_video.py                      # shim — **mock 코드 전체 제거**
├── job_manager.py                      # 변경 최소화, job_queue를 통해 사용
└── main.py                             # 엔드포인트는 media_engine 사용
```

`lecture_service/` 는 Phase A에서 **변경하지 않는다**. Phase B4(비디오 능력)에서 media_engine으로 마이그레이션한다.

## 4. 컴포넌트 상세

### 4.1 `catalog.py`
워크플로우 메타데이터의 단일 출처. 모든 사용처는 여기를 거친다.

```python
WORKFLOWS = {
  "image.gen.flux": {
    "template": "image_gen_flux.json.j2",
    "params": {
      "prompt": (str, ...),
      "negative": (str, ""),
      "width": (int, 1024),
      "height": (int, 1024),
      "steps": (int, 20),
      "seed": (int, None),
    },
    "models_required": [
      "diffusion_models/FLUX1/flux1-dev.safetensors",
      "vae/ae.safetensors",
      "text_encoders/t5xxl_fp8_e4m3fn.safetensors",
      "text_encoders/clip_l.safetensors",
    ],
    "output_node": "9",
    "vram_class": "heavy",       # heavy → vLLM swap 필요
    "timeout_sec": 300,
  },
  "image.gen.zimage_turbo": { ..., "vram_class": "light", "timeout_sec": 60 },
  "image.edit.qwen":        { ..., "vram_class": "heavy", "timeout_sec": 300 },
  "video.i2v.wan22":        { ..., "vram_class": "heavy", "timeout_sec": 600 },
  "video.s2v.wan22":        { ..., "vram_class": "heavy", "timeout_sec": 900 },
}

def get(workflow_id: str) -> dict: ...
def validate(meta: dict, params: dict) -> dict:
    """타입 검사, 기본값 채움, 필수 누락 시 ValueError"""
def check_models_present(meta: dict) -> list[str]:
    """누락 모델 파일명 리스트 반환 (시작 시 호출)"""
```

`vram_class`:
- `light` = ComfyUI VRAM 사용량이 ~10GB 이하로 vLLM과 공존 가능 (예: z_image_turbo)
- `heavy` = vLLM 일시 중단 필요 (Flux dev, Qwen Edit, Wan2.2 14B 등)

### 4.2 `runner.py`

```python
async def run(workflow_id: str, **params) -> Path:
    meta = catalog.get(workflow_id)
    validated = catalog.validate(meta, params)

    async with gpu_arbiter.acquire(meta["vram_class"]):
        prompt_json = render_template(meta["template"], validated)
        prompt_id = await comfyui_client.submit(prompt_json)
        output_file = await comfyui_client.wait_and_fetch(
            prompt_id,
            output_node=meta["output_node"],
            timeout=meta["timeout_sec"],
        )
        return _copy_to_results(output_file, workflow_id)
```

Jinja 렌더링은 워크플로우 JSON을 텍스트로 다룬다(템플릿 변수가 JSON 문자열 내부에 있을 때 escape 필요). `prompt | tojson` 필터로 안전하게 직렬화.

### 4.3 `gpu_arbiter.py`

```python
_lock = asyncio.Lock()        # process-wide heavy 잡 직렬화
_vllm_state = "running"       # "running" | "paused" | "restarting"

@asynccontextmanager
async def acquire(vram_class: str):
    if vram_class == "heavy":
        async with _lock:
            await _pause_vllm()
            try:
                yield
            finally:
                asyncio.create_task(_resume_vllm())  # background restart
    else:
        yield  # light 잡: lock 불필요

async def _pause_vllm():
    global _vllm_state
    _vllm_state = "restarting"
    await run_subprocess("docker", "stop", "vllm-server", timeout=15)
    _vllm_state = "paused"

async def _resume_vllm():
    global _vllm_state
    _vllm_state = "restarting"
    await run_subprocess("docker", "start", "vllm-server", timeout=15)
    # /v1/models 헬스체크
    for _ in range(90):
        if await _vllm_healthy():
            _vllm_state = "running"
            return
        await asyncio.sleep(1)
    _vllm_state = "unhealthy"  # 모니터링에서 알림

def vllm_available() -> bool:
    return _vllm_state == "running"
```

`llm_service.generate_text` 는 호출 직전 `vllm_available()` 확인 → False면 `HTTPException(503, "LLM 일시 정지 중 — GPU 미디어 작업 진행. 30~60초 후 재시도")` 반환.

### 4.4 `window.py`

```python
async def chunk_audio_fixed(path, chunk_sec=30, overlap_sec=5) -> list[Path]
async def chunk_audio_smart(path, target_range=(4, 8), silence_aware=True) -> list[Path]
    # lecture_service의 silence 기반 분할 일반화
async def extract_last_frame(video_path) -> Path
async def concat_videos(paths, output_path, overlap_frames=0) -> Path
    # overlap_frames > 0이면 ffmpeg xfade 사용
async def crossfade_audio(paths, overlap_ms=300, output_path=None) -> Path
async def get_media_duration(path) -> float
```

외부 의존성: `ffmpeg`(시스템), `pydub`(이미 사용 중). lecture_service의 `slice_audio` 로직은 `chunk_audio_smart`로 이전.

### 4.5 `job_queue.py`

```python
async def submit(job_type: str, payload: dict, runner_coro) -> str:
    job_id = job_manager.create_job(job_type, payload)
    asyncio.create_task(_run(job_id, runner_coro))
    return job_id

async def _run(job_id, coro):
    try:
        job_manager.update_job(job_id, "processing")
        result = await coro
        job_manager.update_job(job_id, "completed", result=result)
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))
        _dump_error_log(job_id, e)
```

기존 `job_manager.py`의 dict + json 파일 영속화는 그대로 유지. 직렬화는 `gpu_arbiter`의 lock이 담당하므로 별도 큐 백엔드(Redis 등)는 불필요.

### 4.6 `comfyui_client.py`
HTTP I/O 캡슐화. 기존 `media_image.py`/`orchestrator.py`에 흩어진 ComfyUI 호출 로직 통합.

```python
async def submit(prompt_json: dict) -> str  # returns prompt_id
async def wait_and_fetch(prompt_id, output_node, timeout) -> Path
async def upload_image(local_path, filename) -> dict
async def upload_audio(local_path, filename) -> dict
```

### 4.7 미디어 shim

`media_image.py`, `media_video.py`, `media_audio.py`를 다음과 같이 슬림화:

```python
# media_image.py
from media_engine import runner

async def generate_image(prompt: str, workflow="zimage_turbo", **kw) -> Path:
    workflow_id = f"image.gen.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, **kw)
```

`media_video.py`의 **mock 구현은 전면 삭제**하고 `runner.run("video.i2v.wan22", ...)` 호출로 대체.

## 5. 데이터 흐름

```
client
  └─POST /api/media/image (prompt, workflow)
      └─ main.py 엔드포인트
          └─ job_queue.submit("media_image", payload, runner=media_image.generate(...))
              └─ background task
                  └─ media_image.generate
                      └─ media_engine.runner.run("image.gen.flux", prompt=...)
                          └─ gpu_arbiter.acquire("heavy")
                              ├─ docker stop vllm-server (~10s)
                              └─ ComfyUI POST /prompt → poll /history → fetch
                          └─ acquire 종료: background docker start vllm-server + health
                      ← Path("results/image_...png")
                  ← job_manager.update_job("completed", result_url)
          ← {"job_id": ...}
  └─GET /api/jobs/{job_id} (poll)
  └─GET /api/results/{filename}
```

## 6. 에러 처리

| 실패 유형 | 대응 |
|---|---|
| ComfyUI 워크플로우 노드 에러 | `/history` error 필드 추출 → 사용자 친화 메시지로 매핑 (예: OOM → "해상도를 낮춰주세요") |
| ComfyUI HTTP 타임아웃 (> `timeout_sec`) | 컨테이너 헬스 체크, 1회 자동 재시도, 실패 시 잡 failed |
| vLLM resume 실패 (>90s) | 잡은 완료 처리, vLLM 상태 `unhealthy`로 마킹, `/api/health/vllm` 엔드포인트에 노출 |
| 디스크 잔여 <20GB | 신규 잡 거부 (HTTP 429 + 안내 메시지) |
| 필요 모델 누락 | catalog `check_models_present()` 단계에서 거부, 누락 파일 목록 응답 |
| 동시 heavy 잡 N개 | 자동 직렬화 (gpu_arbiter lock), 사용자에게 "대기 중 N번째" 정보 노출 |

실패 잡은 `results/_errors/{job_id}.log`에 (a) Python traceback, (b) ComfyUI history JSON dump, (c) docker logs vllm-server 최근 100줄 저장. 30일 보관.

## 7. 테스트 계획

### Unit (CI 가능, GPU 불필요)
- `test_catalog.py`
  - 모든 워크플로우 메타데이터의 형식 일관성
  - `validate` 의 타입 강제·기본값·필수 누락 케이스
- `test_runner_mock.py`
  - ComfyUI 클라이언트와 GPU arbiter를 mock하여 `run()` 전체 흐름
- `test_gpu_arbiter.py`
  - docker subprocess mock, lock 직렬성, vLLM 상태 전이
- `test_window.py`
  - 짧은 sample 오디오/비디오 (소스: pydub silent, ffmpeg testsrc)로 청크/오버랩 경계 케이스

### Integration (실제 GPU 필요, 수동)
1. `z_image_turbo` 1024² 8 steps → < 5초 완료, PNG 검증
2. `flux dev` 1024² 20 steps → vLLM stop/start 한 사이클, PNG 검증
3. `wan2.2 i2v` 5초 청크 1건 → MP4 검증, `extract_last_frame` 동작
4. 동시 잡 2건 큐잉 → 순차 처리, 잡 ID 별 결과 검증

### Smoke (배포 직전)
- 기존 6개 엔드포인트가 mock 코드 없이 정상 응답
- vLLM swap 직후 LLM 호출 시 503 → 60초 후 자동 200

## 8. 완료 정의 (Definition of Done)

1. **mock 제거**: `media_video.py`에서 `MOCK` 주석 및 색깔 블록 생성 코드 전부 삭제, 실제 Wan2.2 i2v 호출
2. **공용 엔진 동작**: `media_engine.runner.run()` 으로 image/video/image-edit 워크플로우가 모두 실행됨
3. **GPU 자원 보호**: heavy 잡 실행 중 LLM 호출이 503으로 깔끔히 거부되고, 잡 종료 후 자동 복구
4. **잡 큐 직렬화**: 동시 heavy 잡 N건이 줄세워 처리되며 클라이언트가 진행상황을 폴링 가능
5. **테스트 통과**: unit ≥ 10 케이스, integration 시나리오 4건, smoke 6개 엔드포인트
6. **회귀 없음**: 기존 `/api/llm/chat`, `/api/vlm/analyze`, `/api/audio/stt`, `/api/ppt/generate`는 변경 없이 정상 동작
7. **lecture_service**: Phase A에서는 손대지 않음 (Phase B4에서 통합)

## 9. 범위 외 (Phase B/C로 이연)

- 이미지/오디오/음성/비디오 능력 자체의 신규/강화 (Phase B1~B4)
- 음악·음성·비디오의 긴 길이 생성 워크플로우 자체 (window.py의 유틸은 만들되, 실제 호출 시나리오는 B 단계에서)
- 멀티모달 라우터 및 통합 UI (Phase C)
- 추가 모델 다운로드 (대부분 이미 존재; 누락 발견 시 그때 추가)
- 잡 큐 백엔드 교체 (Redis/Celery 등) — 단일 머신에서는 불필요

## 10. 위험 및 완화

| 위험 | 완화 |
|---|---|
| docker stop이 vLLM의 in-flight 요청을 끊을 수 있음 | acquire 직전 LLM 요청은 502/503 후 클라이언트 재시도로 처리, 신규 요청은 vllm_state로 차단 |
| 디스크 잔여 132GB → 무빙윈도우 중간 산출물 축적 | 각 잡 완료 후 tmp 정리, results는 7일 후 자동 삭제 (별도 cronjob) |
| Wan2.2 14B 첫 실행 시 모델 로딩 ~60초 | 첫 호출에 워밍업 마킹, 사용자에게 "초기화 중" 상태 노출 |
| ComfyUI 컨테이너 자체 OOM/크래시 | `docker restart comfyui` 후 1회 재시도, 그래도 실패면 fail-fast |

---

## 부록 — 디렉토리 매핑 요약

| 기존 | 변경 후 |
|---|---|
| `media_image.py` 의 워크플로우 dict | `media_engine/workflows/image_gen_*.json.j2` + `catalog.py` |
| `media_video.py` 의 MOCK 코드 | **삭제**, `media_engine/runner.run("video.i2v.wan22")` |
| `lecture_service/workflows.py` | Phase A에서는 유지, Phase B4에서 `media_engine/workflows/`로 통합 |
| `job_manager.py` | 유지, `media_engine/job_queue.py` 가 래핑 |
| `llm_service.py` | `vllm_available()` 체크 1줄 추가 |
