# Phase B Core — Multimodal Task Router Design

**Date:** 2026-05-23  
**Scope:** `unified_ai_service` Phase B의 중심 계층. 사용자의 자연어 요청과 첨부 파일을 받아 이미지, 오디오, 음성, 비디오, PPT, 분석 도구를 조합 실행한다.

## 1. 목표

사용자가 요청 내용을 미디어 종류별 API로 직접 쪼개지 않아도, `/api/multimodal/execute`에 자연어 지시와 파일을 제출하면 시스템이 실행 가능한 작업 계획을 만들고 순차 실행한다.

핵심 목표:

- 이미지, 오디오, 음성, 비디오, 문서 작업을 한 요청 안에서 조합한다.
- 생성, 편집, 변환, 수정, 분석 작업을 공통 `MediaPlan` 형식으로 표현한다.
- 긴 음성, 긴 음악, 긴 비디오 작업은 문맥을 보존하는 window 기반 실행 전략을 갖는다.
- 발표자료와 강의자료에는 대본, TTS, 슬라이드, 립싱크 영상으로 이어지는 end-to-end 경로를 제공한다.
- 로컬 모델을 기본값으로 쓰되, 품질이 중요한 긴 TTS는 ElevenLabs를 선택형 provider로 사용한다.
- 외부 API key가 없거나 외부 API가 실패해도 로컬 fallback으로 작업을 계속할 수 있다.

## 2. 현재 상태

이미 구현된 기반:

- `media_engine.runner.run(workflow_id, **params)`로 ComfyUI 워크플로우 실행.
- `media_engine.job_queue.submit(...)`로 장시간 작업을 job으로 등록.
- `media_image.py`: 이미지 생성, Qwen edit, FLUX ControlNet, Qwen inpaint.
- `media_video.py`: Wan2.2 i2v 긴 비디오 생성, 편집, shorts, 키프레임 분석.
- `media_audio.py`: MusicGen-small 음악, F5-TTS + SFX, 단순 긴 음악 chunk/crossfade.
- `main.py`: 미디어별 endpoint와 `/api/jobs/{job_id}` job polling.
- `llm_service.py`: vLLM 기반 텍스트 생성과 이미지 분석.
- `ppt_service.py`: LLM slide JSON을 `.pptx`로 저장.

현재 부족한 부분:

- 자연어 요청을 미디어 도구 호출로 분해하는 중심 라우터가 없다.
- 여러 step의 중간 산출물을 다음 step 입력으로 넘기는 공통 실행기가 없다.
- TTS provider가 로컬 F5-TTS 하나뿐이고 긴 대본 품질 fallback이 없다.
- 긴 미디어 작업의 문맥, 화자, 객체, 사운드 추적 metadata가 표준화되어 있지 않다.
- 립싱크는 워크플로우 템플릿만 준비되어 있고 강의자료 pipeline에는 아직 연결되지 않았다.

## 3. 외부 기술 선택

### 3.1 ElevenLabs TTS

사용처:

- 긴 강의 내레이션.
- 자연스러운 대화형 TTS.
- 사용자가 “더 자연스럽게”, “전문 성우처럼”, “감정 있게” 같은 품질 지시를 준 경우.

설계:

- 환경변수 `ELEVENLABS_API_KEY`가 있으면 `elevenlabs` provider를 사용할 수 있다.
- 없으면 `local_f5` provider를 사용한다.
- `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`를 선택적으로 지원한다.
- 기본 model id는 안정적인 long-form 품질을 우선해 `eleven_multilingual_v2`로 둔다.
- streaming API는 초기 구현에서 네트워크 응답을 파일로 저장하는 방식으로 사용한다.

### 3.2 WhisperX + pyannote

사용처:

- 긴 오디오/비디오의 STT.
- word-level timestamp.
- speaker diarization.
- 강의자료에서 화자별 대본, 자막, 장면 매칭 metadata 생성.

초기 구현에서는 라우터 capability로 선언하고, 실행 hook은 fallback 메시지를 반환한다. 실제 모델 설치와 diarization pipeline은 Phase B3에서 별도 작업으로 붙인다.

### 3.3 YOLO Tracking + SAM 2

사용처:

- 비디오 객체 추적.
- 인물/객체별 crop, blur, remove, replace, follow-camera.
- channel tracking: 동일 객체 id를 긴 영상 chunk 사이에서 이어받는 metadata 생성.

초기 구현에서는 capability와 plan schema에 `track_target`, `track_id`, `track_strategy` 필드를 정의한다. 실제 실행은 Phase B4에서 YOLO BoT-SORT/ByteTrack을 먼저 붙이고, 정밀 마스크가 필요한 편집은 SAM 2를 붙인다.

### 3.4 Wav2Lip 계열

사용처:

- 발표자 얼굴 영상 + TTS 음성의 립싱크.
- 강의자료, 발표자료, 교육 영상.

초기 구현에서는 `video.lipsync` action을 plan schema에 포함하고, 기존 `video.s2v.wan22` 워크플로우와 외부 Wav2Lip runner 중 하나로 실행할 수 있게 인터페이스를 분리한다. 실제 구현 우선순위는 Phase B4다.

## 4. 아키텍처

새 파일:

- `unified_ai_service/multimodal_models.py`
  - `MediaAsset`, `MediaPlan`, `MediaStep`, `StepResult` dataclass.
  - JSON 직렬화와 검증.
- `unified_ai_service/media_capabilities.py`
  - 시스템이 지원하는 action 목록과 입력/출력 타입.
  - LLM planner가 참고할 compact capability prompt 생성.
- `unified_ai_service/multimodal_router.py`
  - 자연어 요청과 첨부 asset을 받아 `MediaPlan` 생성.
  - LLM JSON 결과 파싱, 실패 시 rule-based fallback plan 생성.
- `unified_ai_service/multimodal_executor.py`
  - `MediaPlan`을 step별로 실행.
  - 중간 결과 alias 저장.
  - job result에 plan, step result, final result를 저장.
- `unified_ai_service/voice_providers.py`
  - `synthesize_speech(...)` 공통 함수.
  - `local_f5`와 `elevenlabs` provider.
  - 긴 텍스트 chunking과 provider fallback.

수정 파일:

- `unified_ai_service/main.py`
  - `POST /api/multimodal/execute` 추가.
  - multipart form: `instruction`, `files`, optional `quality`, optional `preferred_voice_provider`.
- `unified_ai_service/media_audio.py`
  - TTS 호출을 `voice_providers.synthesize_speech`로 위임.
- `unified_ai_service/requirements.txt`
  - `httpx>=0.27` 추가. ElevenLabs 호출은 직접 HTTP로 구현해 SDK churn을 피한다.
- `unified_ai_service/media_engine/README.md`
  - Multimodal endpoint와 provider 설정 문서화.

## 5. MediaPlan Schema

Plan은 항상 다음 구조다.

```json
{
  "version": "1",
  "goal": "사용자 요청 요약",
  "quality": "standard",
  "steps": [
    {
      "id": "step_1",
      "action": "voice.tts",
      "inputs": {
        "text": "강의 대본",
        "provider": "auto",
        "voice": "default"
      },
      "outputs": {
        "audio": "lecture_voice"
      }
    }
  ],
  "final": {
    "primary": "lecture_voice",
    "format": "audio"
  }
}
```

Rules:

- `version`은 `"1"`만 허용한다.
- `steps[*].id`는 plan 안에서 unique 해야 한다.
- `action`은 `media_capabilities.py`에 등록된 값만 허용한다.
- `inputs`는 primitive, uploaded asset alias, 이전 step output alias만 참조한다.
- `outputs`의 alias는 unique 해야 한다.
- executor는 step 순서대로만 실행한다. 병렬 실행은 Phase C 이전에는 도입하지 않는다.
- planner가 알 수 없는 action을 반환하면 plan validation에서 실패하고 fallback plan을 생성한다.

초기 action 목록:

- `text.generate`: LLM 텍스트 생성.
- `ppt.generate`: PPT 생성.
- `image.generate`: Z-Image 또는 FLUX 이미지 생성.
- `image.edit`: Qwen image edit.
- `image.control`: FLUX ControlNet.
- `image.inpaint`: Qwen inpaint.
- `image.analyze`: VLM 이미지 분석.
- `audio.music`: 음악 생성.
- `audio.transcribe`: STT.
- `voice.tts`: TTS.
- `video.generate`: Wan2.2 i2v 긴 비디오.
- `video.edit`: moviepy 기반 편집.
- `video.analyze`: 키프레임 기반 분석.
- `video.shorts`: shorts 생성.
- `video.lipsync`: 강의/발표 립싱크. 초기에는 unsupported로 명확히 실패하고 plan에는 남긴다.
- `package.bundle`: 여러 산출물을 JSON manifest로 묶는다.

## 6. 라우팅 전략

### 6.1 LLM Planner

`multimodal_router.plan(...)`은 다음 prompt를 LLM에 전달한다.

- 사용자 instruction.
- 첨부 asset 목록: alias, mime type, filename, saved path.
- capability 요약.
- JSON schema 규칙.
- 비용 정책: 외부 API는 `quality=high` 또는 TTS 자연스러움 요구가 있을 때 우선 사용.

LLM 응답은 JSON object만 허용한다. markdown code fence가 오면 제거 후 parse한다.

### 6.2 Rule-Based Fallback

LLM이 unavailable이거나 invalid JSON을 반환하면 다음 rule을 적용한다.

- 파일 없이 “이미지/그림/사진 생성” → `image.generate`.
- 이미지 파일 + “편집/수정/바꿔/제거” → `image.edit`.
- 마스크 파일 2개 이상 + “제거/inpaint” → `image.inpaint`.
- 오디오 파일 + “텍스트/자막/전사” → `audio.transcribe`.
- 텍스트 + “음성/내레이션/TTS/읽어줘” → `voice.tts`.
- 이미지 파일 + “영상/비디오/움직여” → `video.generate`.
- 비디오 파일 + “분석/설명” → `video.analyze`.
- “발표자료/PPT/슬라이드” → `ppt.generate`.
- 여러 결과가 필요한 요청 → `package.bundle`을 마지막 step으로 추가한다.

Fallback은 완벽한 이해가 아니라 안전한 실행을 목표로 한다. 모호한 요청은 `text.generate`로 해석 결과와 필요한 입력을 설명한다.

## 7. 긴 미디어 전략

### 7.1 긴 음성

긴 TTS는 문단 단위로 chunk한다.

- 기본 chunk limit: 2500 characters.
- ElevenLabs provider: provider 제한보다 낮은 안전 limit 사용.
- Local F5 provider: 문장 경계 기준으로 더 짧게 chunk한다.
- 각 chunk는 `context_before`, `context_after`를 metadata로 받는다.
- 출력은 pydub으로 이어붙이고 문단 경계에는 짧은 silence를 넣는다.
- 결과 manifest에는 chunk index, text range, provider, output path를 남긴다.

### 7.2 긴 음악

현재 `generate_long_music`은 10초 chunk + crossfade다. 라우터는 이 기능을 그대로 호출하되, plan metadata에 `continuity_prompt`를 추가한다.

Phase B2에서 개선:

- 더 큰 MusicGen 모델 또는 ComfyUI audio workflow.
- 이전 chunk의 audio embedding이나 summary를 다음 prompt에 반영.

### 7.3 긴 비디오

현재 `generate_long_video`는 마지막 프레임을 다음 chunk seed로 사용한다. 라우터는 이 기능을 `video.generate`로 호출한다.

Phase B4에서 개선:

- shot plan 생성.
- 객체 track id 유지.
- SAM 2 mask tracking.
- audio timeline과 visual shot timeline 동기화.

### 7.4 Channel Tracking

초기 schema는 channel tracking metadata를 담을 수 있게 한다.

```json
{
  "channels": {
    "speakers": [{"id": "speaker_1", "label": "강사"}],
    "objects": [{"id": "object_1", "label": "발표자"}],
    "audio_events": [{"id": "event_1", "label": "박수"}]
  }
}
```

초기 구현에서는 metadata를 보존하고 전달만 한다. 실제 추적기는 Phase B3/B4에서 붙인다.

## 8. ElevenLabs 설정

환경변수:

- `ELEVENLABS_API_KEY`: API key. 없으면 ElevenLabs provider는 unavailable.
- `ELEVENLABS_VOICE_ID`: 기본 voice id. 없으면 요청이 `elevenlabs`를 강제할 때 명확한 오류를 반환하고, `auto`일 때 local fallback.
- `ELEVENLABS_MODEL_ID`: 기본 `eleven_multilingual_v2`.
- `ELEVENLABS_OUTPUT_FORMAT`: 기본 `mp3_44100_128`.

Provider 선택:

- `provider=local_f5`: 로컬만 사용.
- `provider=elevenlabs`: ElevenLabs만 사용. key/voice 없으면 실패.
- `provider=auto`: high quality TTS 요청이면 ElevenLabs 시도 후 local fallback.

비용 통제:

- job result에 provider와 estimated character count를 기록한다.
- 초기 구현은 비용 한도 계산을 하지 않는다.
- Phase C에서 사용자별 quota와 승인 UI를 붙인다.

## 9. Endpoint

`POST /api/multimodal/execute`

Form fields:

- `instruction: str` required.
- `quality: str` optional, one of `draft`, `standard`, `high`. default `standard`.
- `preferred_voice_provider: str` optional, one of `auto`, `local_f5`, `elevenlabs`. default `auto`.
- `files: list[UploadFile]` optional.

Response:

```json
{
  "job_id": "abc123"
}
```

Job result on completion:

```json
{
  "plan": {},
  "steps": [
    {
      "id": "step_1",
      "action": "voice.tts",
      "status": "completed",
      "result": {
        "audio": "/api/results/tts_x.wav"
      }
    }
  ],
  "final": {
    "type": "audio",
    "url": "/api/results/tts_x.wav"
  }
}
```

## 10. Error Handling

- Plan parse failure: fallback planner 사용.
- Plan validation failure: fallback planner 사용. fallback도 실패하면 job failed.
- Step failure: job failed, failing step id/action/error 기록.
- Unsupported action: job failed with `unsupported_action`, unless action is optional.
- ElevenLabs network failure with `provider=auto`: local fallback.
- ElevenLabs failure with `provider=elevenlabs`: job failed.
- Missing uploaded asset alias: plan validation failure.
- Output file missing after a step: step failure.
- vLLM unavailable: planner fallback, executor는 가능한 non-LLM 작업만 수행.

## 11. Security and Privacy

- Uploaded files are saved under existing `UPLOADS_DIR`.
- Result files remain under existing `RESULTS_DIR`.
- External API upload is limited to TTS text for ElevenLabs in the initial implementation.
- Audio/video/image files are not sent to external APIs in Phase B Core.
- The job result records which provider was used.
- API keys are read only from environment variables and never returned in responses.

## 12. Testing Strategy

Unit tests:

- `MediaPlan` validates valid plans.
- `MediaPlan` rejects unknown action, duplicate step id, duplicate output alias.
- Capability prompt includes all supported initial actions.
- Router parses clean JSON from LLM.
- Router strips markdown code fence.
- Router falls back when LLM returns invalid JSON.
- Rule fallback maps common Korean requests to expected actions.
- Executor runs two chained fake steps and passes output alias.
- Executor fails with clear step error.
- Voice provider chooses local when no ElevenLabs key.
- Voice provider uses ElevenLabs when key and voice are present.
- ElevenLabs network failure falls back in `auto`.

Integration smoke:

- `POST /api/multimodal/execute` with “고양이 이미지를 만들어줘” returns job id and eventually image result.
- `POST /api/multimodal/execute` with TTS request and no ElevenLabs key returns local TTS or clear local F5 error.
- `POST /api/multimodal/execute` with image + “이 이미지를 분석해줘” returns text analysis.

GPU-heavy integration remains opt-in and is not run by default.

## 13. Implementation Order

1. Data model and capability registry.
2. Router with LLM JSON parsing and rule fallback.
3. Executor with fake action tests.
4. Voice provider abstraction with ElevenLabs optional API.
5. `main.py` endpoint.
6. Connect executor actions to existing media modules.
7. README update.
8. Smoke tests.

## 14. Non-Goals for Phase B Core

- Full YOLO/SAM 2 object tracking implementation.
- Full WhisperX/pyannote diarization implementation.
- Full Wav2Lip installation and model download.
- User quota, billing, or approval UI for paid API usage.
- Parallel execution graph scheduler.
- Browser UI redesign.

These are explicit follow-up phases after the router can represent and dispatch multimodal work.

## 15. Follow-Up Phase Map

- **Phase B1b:** FLUX Lightning LoRA and faster image generation.
- **Phase B1c:** Janus-Pro or equivalent multimodal image/video understanding.
- **Phase B2:** long music/audio generation quality, audio continuity, sound event timeline.
- **Phase B3:** long STT, diarization, speaker tracking, subtitle generation, ElevenLabs polish.
- **Phase B4:** video tracking, SAM 2 masks, Wav2Lip/lecture lip-sync, Wan2.2 multi-stage.
- **Phase C:** user-facing multimodal UI, quota/approval, interactive plan review.

## 16. Acceptance Criteria

Phase B Core is complete when:

- `/api/multimodal/execute` accepts natural language instructions and files.
- At least image generation, image analysis, TTS, PPT generation, video analysis, and package actions can be planned and executed through the router.
- Invalid LLM planner output does not break the endpoint; fallback plan is used.
- ElevenLabs is optional and never required for local operation.
- Job result includes the executed plan and every step result.
- Unit tests pass without GPU or external API access.
- README documents endpoint, environment variables, and known follow-up phases.

## 17. References

- ElevenLabs Text to Speech and streaming API documentation.
- Ultralytics YOLO tracking documentation for BoT-SORT and ByteTrack.
- WhisperX long-form transcription with word-level timestamps and diarization.
- pyannote.audio speaker diarization.
- Meta Segment Anything Model 2 for video object segmentation and tracking.
- Wav2Lip paper and implementation family for audio-driven lip synchronization.
