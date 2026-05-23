# media_engine

`unified_ai_service`의 공용 미디어 엔진. Phase A 산출물 (2026-05-23).

## 구성

- **catalog.py** — 워크플로우 메타데이터 단일 출처 (7 워크플로우, 파라미터·모델·VRAM class·timeout)
- **workflows/** — ComfyUI API JSON + Jinja 템플릿 (zimage_turbo / flux / flux control / qwen_edit / qwen_inpaint / wan22 i2v / wan22 s2v)
- **comfyui_client.py** — ComfyUI HTTP 클라이언트 (submit / poll history / upload, `ComfyUIError`)
- **gpu_arbiter.py** — heavy 잡 직렬화 + vLLM `docker stop/start` swap (`acquire(vram_class)`, `vllm_available()`, `state()`)
- **window.py** — 무빙윈도우 유틸 (`chunk_audio_fixed/smart`, `extract_last_frame`, `concat_videos`, `crossfade_audio_segments`, `get_media_duration`)
- **runner.py** — 통합 실행 진입점 `run(workflow_id, **params)`
- **job_queue.py** — `job_manager` 위 비동기 잡 등록 + `_errors/{job_id}.log` 덤프

## 사용 예

```python
from media_engine import runner

# 이미지 생성 (light, vLLM과 공존)
img = await runner.run("image.gen.zimage_turbo", prompt="a cat", steps=4)

# 고품질 이미지 생성 (heavy, vLLM swap)
flux = await runner.run("image.gen.flux", prompt="a cinematic mountain lake")

# 이미지 편집 (heavy, vLLM swap)
edited = await runner.run("image.edit.qwen", prompt="make it night", image_name="uploaded.png")

# 5초 비디오 생성 (heavy)
vid = await runner.run("video.i2v.wan22", prompt="gentle motion", image_name="seed.png", frames=81)
```

새 워크플로우 추가 절차는 [`workflows/README.md`](workflows/README.md) 참조.

## 테스트

```bash
# 단위 (CI 가능, GPU 불필요)
./venv/bin/python -m pytest media_engine/tests/ -v
# → 52 passed, 6 skipped (--integration 필요)

# 통합 (실제 GPU + ComfyUI + vLLM 필요, ~10-15분)
./venv/bin/python -m pytest media_engine/tests/integration -v --integration -s --timeout=1800
```

## 엔드포인트 (`main.py` 노출)

| Method | Path | 워크플로우 |
|---|---|---|
| POST | `/api/media/image` | image.gen.zimage_turbo (default) |
| POST | `/api/media/image?workflow=flux` | image.gen.flux (FLUX dev fp8) |
| POST | `/api/media/image/edit` | image.edit.qwen (신규 — Phase A 산출) |
| POST | `/api/media/image/control` | image.ctrl.flux_union (FLUX-ControlNet Union) |
| POST | `/api/media/image/inpaint` | image.inpaint.qwen (Qwen + 마스크) |
| POST | `/api/media/music` | MusicGen-small. `duration > 30` 이면 `generate_long_music` (moving window) |
| POST | `/api/media/tts` | F5-TTS + SFX 태그 (기존) |
| POST | `/api/media/video/gen` | video.i2v.wan22 + moving window |
| POST | `/api/media/video/edit` | moviepy (오디오 덧입히기 / 이미지 append) |
| POST | `/api/media/video/shorts` | 자르기 + 9:16 세로 크롭 |
| POST | `/api/media/video/analyze` | 다중 키프레임 VLM 종합 |
| GET | `/api/health/vllm` | `{"state": "...", "available": bool}` |

각 POST는 `{"job_id": "..."}` 반환. `GET /api/jobs/{job_id}` 로 진행/결과 폴링.

## Phase B1a 산출 (2026-05-23)

- 신규 워크플로우 3개:
  - `image.gen.flux` — FLUX dev fp8 text2img (heavy, ~5분)
  - `image.ctrl.flux_union` — FLUX-ControlNet-Union-Pro (canny/openpose/depth/scribble)
  - `image.inpaint.qwen` — Qwen Image Edit + 마스크 inpaint
- 신규 엔드포인트:
  - `POST /api/media/image/control` (control_image + control_type)
  - `POST /api/media/image/inpaint` (image + mask)
  - `POST /api/media/image?workflow=flux` (기존 endpoint 옵션 추가)
- 디스크 추가: FLUX fp8 (~17GB), FLUX text encoders (`clip_l` ~234MB, `t5xxl_fp8` ~4.6GB), FLUX-ControlNet-Union-Pro (~6GB)
- Custom node: ComfyUI-controlnet-aux (canny/openpose/depth preprocessor)

자동 다운로드: `unified_ai_service/scripts/download_b1a_models.sh`

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
- 안정성 설정:
  - `MULTIMODAL_PLANNER_TIMEOUT`: LLM planner 최대 대기 시간. 기본 `8`초
  - `MULTIMODAL_LLM_STEP_TIMEOUT`: LLM 기반 실행 step 최대 대기 시간. 기본 `20`초
- 로컬 fallback:
  - key가 없으면 `auto` provider는 `local_f5`로 실행
  - `f5-tts_infer-cli`가 없고 ElevenLabs key/voice가 있으면 `auto` provider는 ElevenLabs로 자동 전환
  - ElevenLabs 장애 시 `auto`는 로컬 fallback
  - `preferred_voice_provider=elevenlabs`로 강제했을 때만 ElevenLabs 실패가 job 실패가 됨

## 알려진 한계 (Phase B에서 개선)

- **첫 실행 cold start**: ComfyUI가 모델을 처음 로드할 때 light 워크플로우도 60-90초 걸릴 수 있어 catalog의 `timeout_sec: 90`을 초과할 수 있음. 해결안: `timeout_sec` 상향(180s), 또는 서비스 기동 시 더미 prompt로 워밍업.
- **vLLM resume 타임아웃**: Gemma-4-26B fp8 모델 cold load가 5-7분 걸릴 수 있어 기본 `VLLM_RESUME_TIMEOUT=420`을 사용. 디스크/네트워크 상태에 따라 환경변수로 더 길게 설정 가능.
- **Wan2.2 i2v는 high_noise single-stage**: `unet_low`는 catalog에 등록만 (Phase B4에서 multi-stage 확장).
- **음악 길이 ≤ MusicGen-small chunk**: 30초 이상은 crossfade로 잇지만 음악 일관성은 청크 경계에서 끊김. Phase B2에서 더 큰 모델로 교체.
- **lecture_service 미통합**: Phase A에서는 lecture_service의 Wan2.2 s2v 흐름을 그대로 유지. Phase B4에서 media_engine으로 마이그레이션.

## 관련 문서

- 설계 spec: [`/home/yanus/docs/superpowers/specs/2026-05-23-phase-a-media-engine-design.md`](../../docs/superpowers/specs/2026-05-23-phase-a-media-engine-design.md)
- 구현 계획: [`/home/yanus/docs/superpowers/plans/2026-05-23-phase-a-media-engine.md`](../../docs/superpowers/plans/2026-05-23-phase-a-media-engine.md)

## Phase A 산출 요약

- 18 commits (`65ee2f6..f2ddd1c`)
- `media_engine/` 패키지 (6 modules + 4 workflows + 8 test files)
- `media_video.py` mock 코드 완전 제거 → Wan2.2 i2v 실제 연결
- 41 unit tests passing, 모든 모듈 import OK
- 9개 미디어/health 엔드포인트 등록
- Integration 1차 시험에서 Wan2.2 i2v 실제 mp4 산출 (396KB) + vLLM docker swap 사이클 정상 작동 확인
