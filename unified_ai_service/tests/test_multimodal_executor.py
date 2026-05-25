from pathlib import Path

import pytest

from multimodal_models import MediaPlan
import multimodal_executor


def _plan(raw):
    return MediaPlan.from_dict(
        raw,
        supported_actions=multimodal_executor.SUPPORTED_ACTIONS,
        asset_aliases={"image_1", "audio_1", "file_1"},
    )


@pytest.mark.asyncio
async def test_executor_chains_text_output(monkeypatch):
    async def fake_text(inputs, ctx):
        return {"text": "generated text"}

    async def fake_tts(inputs, ctx):
        assert inputs["text"] == "generated text"
        return {"audio": "/api/results/voice.wav"}

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "text.generate", fake_text)
    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "voice.tts", fake_tts)
    plan = _plan(
        {
            "version": "1",
            "goal": "chain",
            "steps": [
                {"id": "s1", "action": "text.generate", "inputs": {"prompt": "write"}, "outputs": {"text": "script"}},
                {"id": "s2", "action": "voice.tts", "inputs": {"text": "$script"}, "outputs": {"audio": "voice"}},
            ],
            "final": {"primary": "voice", "format": "audio"},
        }
    )

    result = await multimodal_executor.execute_plan(plan, assets=[])

    assert result["final"]["url"] == "/api/results/voice.wav"
    assert result["steps"][1]["status"] == "completed"


@pytest.mark.asyncio
async def test_executor_extracts_uploaded_script_text(tmp_path):
    script = tmp_path / "lecture.md"
    script.write_text("# 강의 대본\n\n안녕하세요. 오늘은 AI를 설명합니다.", encoding="utf-8")
    asset = multimodal_executor.MediaAsset(
        alias="file_1",
        path=str(script),
        mime_type="text/markdown",
        filename="lecture.md",
    )
    plan = _plan(
        {
            "version": "1",
            "goal": "script",
            "steps": [
                {"id": "s1", "action": "text.extract", "inputs": {"file": "file_1"}, "outputs": {"text": "script_text"}},
            ],
            "final": {"primary": "script_text", "format": "text"},
        }
    )

    result = await multimodal_executor.execute_plan(plan, assets=[asset])

    assert "오늘은 AI를 설명합니다" in result["final"]["value"]


@pytest.mark.asyncio
async def test_executor_chains_script_to_voice_and_video_overlay(tmp_path, monkeypatch):
    script = tmp_path / "lecture.txt"
    script.write_text("강의 음성입니다.", encoding="utf-8")
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    assets = [
        multimodal_executor.MediaAsset(alias="file_1", path=str(script), mime_type="text/plain"),
        multimodal_executor.MediaAsset(alias="image_1", path=str(image), mime_type="image/png"),
    ]

    async def fake_tts(inputs, ctx):
        assert inputs["text"] == "강의 음성입니다."
        return {"audio": "/home/yanus/unified_ai_service/results/voice.wav"}

    async def fake_video_generate(inputs, ctx):
        assert inputs["image"] == str(image)
        return {"video": "/home/yanus/unified_ai_service/results/silent.mp4"}

    async def fake_video_edit(inputs, ctx):
        assert inputs["video"] == "/home/yanus/unified_ai_service/results/silent.mp4"
        assert inputs["audio"] == "/home/yanus/unified_ai_service/results/voice.wav"
        return {"video": "/home/yanus/unified_ai_service/results/lecture.mp4"}

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "voice.tts", fake_tts)
    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "video.generate", fake_video_generate)
    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "video.edit", fake_video_edit)
    plan = _plan(
        {
            "version": "1",
            "goal": "lecture.video",
            "steps": [
                {"id": "s1", "action": "text.extract", "inputs": {"file": "file_1"}, "outputs": {"text": "script_text"}},
                {
                    "id": "s2",
                    "action": "voice.tts",
                    "inputs": {"text": "$script_text", "provider": "elevenlabs", "voice": "voice123"},
                    "outputs": {"audio": "voice_audio"},
                },
                {
                    "id": "s3",
                    "action": "video.generate",
                    "inputs": {"image": "image_1", "prompt": "강의 영상"},
                    "outputs": {"video": "silent_video"},
                },
                {
                    "id": "s4",
                    "action": "video.edit",
                    "inputs": {"video": "$silent_video", "audio": "$voice_audio", "prompt": "오디오에 맞춰 합성"},
                    "outputs": {"video": "lecture_video"},
                },
            ],
            "final": {"primary": "lecture_video", "format": "video"},
        }
    )

    result = await multimodal_executor.execute_plan(plan, assets=assets)

    assert result["final"]["url"] == "/api/results/lecture.mp4"


@pytest.mark.asyncio
async def test_executor_chains_tts_file_path_not_public_url(tmp_path, monkeypatch):
    audio = Path("/home/yanus/unified_ai_service/results/voice.wav")

    async def fake_tts(inputs, ctx):
        return {
            "audio": "/api/results/voice.wav",
            "metadata": {
                "path": str(audio),
                "url": "/api/results/voice.wav",
                "provider": "elevenlabs",
            },
        }

    async def fake_video_edit(inputs, ctx):
        assert inputs["audio"] == str(audio)
        return {"video": "/home/yanus/unified_ai_service/results/lecture.mp4"}

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "voice.tts", fake_tts)
    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "video.edit", fake_video_edit)
    plan = _plan(
        {
            "version": "1",
            "goal": "audio-chain",
            "steps": [
                {"id": "s1", "action": "voice.tts", "inputs": {"text": "hello"}, "outputs": {"audio": "voice"}},
                {
                    "id": "s2",
                    "action": "video.edit",
                    "inputs": {"video": "video.mp4", "audio": "$voice"},
                    "outputs": {"video": "lecture"},
                },
            ],
            "final": {"primary": "lecture", "format": "video"},
        }
    )

    result = await multimodal_executor.execute_plan(plan, assets=[])

    assert result["steps"][0]["result"]["audio"] == "/api/results/voice.wav"
    assert result["final"]["url"] == "/api/results/lecture.mp4"


@pytest.mark.asyncio
async def test_executor_records_step_failure(monkeypatch):
    async def fail(inputs, ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(multimodal_executor.ACTION_HANDLERS, "text.generate", fail)
    plan = _plan(
        {
            "version": "1",
            "goal": "bad",
            "steps": [{"id": "s1", "action": "text.generate", "inputs": {"prompt": "x"}, "outputs": {"text": "out"}}],
            "final": {"primary": "out", "format": "text"},
        }
    )

    with pytest.raises(multimodal_executor.StepExecutionError, match="s1"):
        await multimodal_executor.execute_plan(plan, assets=[])


@pytest.mark.asyncio
async def test_path_result_is_converted_to_url(tmp_path):
    out = tmp_path / "x.png"
    out.write_bytes(b"png")

    assert multimodal_executor._normalize_result({"image": str(out)})["image"] == str(out)

    result_path = Path("/home/yanus/unified_ai_service/results/x.png")
    normalized = multimodal_executor._normalize_result({"image": result_path})

    assert normalized["image"] == "/api/results/x.png"


@pytest.mark.asyncio
async def test_text_generate_handler_times_out(monkeypatch):
    import asyncio
    import llm_service

    async def slow_generate_text(prompt, system_prompt):
        await asyncio.sleep(1)
        return "late"

    monkeypatch.setattr(llm_service, "generate_text", slow_generate_text)
    monkeypatch.setenv("MULTIMODAL_LLM_STEP_TIMEOUT", "0.01")

    with pytest.raises(TimeoutError):
        await multimodal_executor._handle_text_generate({"prompt": "x"}, multimodal_executor.ExecutionContext([], "draft"))
