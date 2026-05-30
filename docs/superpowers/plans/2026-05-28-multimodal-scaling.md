# Plan: Extreme Multimodal Scaling & Modularization

## Goal
Overhaul `unified_ai_service` to support complex video formats (30+ min PPT-synced lectures, ads, drama), advanced analysis, and modularize the codebase to prevent monolithic errors. Run a 50-case stress test to validate.

## Constraints
- Maximize 64GB VRAM. Heavy tasks must pause vLLM (`gpu_arbiter.acquire("heavy")`).
- No paid services except ElevenLabs.
- Code modularization is mandatory.

## Task 1: Modularize `media_video.py` and `multimodal_executor.py`
- Create `unified_ai_service/media_pipeline/` module.
- Move video logic into `media_pipeline/video_director.py`, `media_pipeline/lecture_sync.py`, `media_pipeline/video_editor.py`.
- Break down `multimodal_executor.py` into `handlers_image.py`, `handlers_video.py`, `handlers_audio.py`, `handlers_doc.py`.

## Task 2: Advanced Lecture Pipeline (`video.lecture.pro`)
- Accept: Presenter Image, Full Script, PPT Topic/File, BGM prompt.
- Logic:
  1. Generate PPT slides as images.
  2. Split script per slide (using LLM).
  3. Generate TTS per slide.
  4. Generate Lip-sync for presenter (optional PiP overlay on PPT).
  5. Merge audio + PPT + Presenter + BGM using `moviepy`.

## Task 3: Creative Video Pipeline (Drama, Ad, Animation) (`video.storyboard`)
- Accept: Creative Prompt.
- Logic:
  1. LLM breaks prompt into 3-4 scenes (Storyboard).
  2. For each scene: generate base image -> generate i2v video (Wan2.2).
  3. Generate matching BGM/Voiceover.
  4. Concatenate scenes.

## Task 4: Enhance Multimodal Router
- Map prompts targeting "드라마" (drama), "광고" (ad), "애니메이션" (animation) to `video.storyboard`.
- Map "긴 강의", "PPT와 함께" to `video.lecture.pro`.

## Task 5: 50-Case Stress Test & Admin Report
- Update `run_50_tests.py` to target these new advanced endpoints.
- Update `admin.html` to load `test_report.json` and display it in a neat table.

## Execution
I will perform these tasks systematically, prioritizing safe imports and preventing `NameError` or `ImportError`.
