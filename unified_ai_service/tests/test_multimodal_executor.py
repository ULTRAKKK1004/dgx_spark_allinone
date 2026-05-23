from pathlib import Path

import pytest

from multimodal_models import MediaPlan
import multimodal_executor


def _plan(raw):
    return MediaPlan.from_dict(
        raw,
        supported_actions=multimodal_executor.SUPPORTED_ACTIONS,
        asset_aliases={"image_1", "audio_1"},
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
