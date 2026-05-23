import pytest

from multimodal_models import MediaAsset
import multimodal_router


@pytest.mark.asyncio
async def test_router_parses_clean_llm_json(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return """
        {"version":"1","goal":"tts","quality":"high","steps":[{"id":"step_1","action":"voice.tts","inputs":{"text":"hello"},"outputs":{"audio":"voice"}}],"final":{"primary":"voice","format":"audio"}}
        """

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("읽어줘", assets=[], quality="high")

    assert plan.steps[0].action == "voice.tts"
    assert plan.quality == "high"


@pytest.mark.asyncio
async def test_router_strips_markdown_fence(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return """```json
        {"version":"1","goal":"image","steps":[{"id":"step_1","action":"image.generate","inputs":{"prompt":"cat"},"outputs":{"image":"img"}}],"final":{"primary":"img","format":"image"}}
        ```"""

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("고양이 이미지", assets=[])

    assert plan.steps[0].action == "image.generate"


@pytest.mark.asyncio
async def test_router_falls_back_on_invalid_json(monkeypatch):
    async def fake_generate_text(prompt, system_prompt):
        return "not json"

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", fake_generate_text)

    plan = await multimodal_router.plan_request("강의 음성으로 읽어줘", assets=[])

    assert plan.steps[0].action == "voice.tts"


@pytest.mark.asyncio
async def test_router_falls_back_when_llm_planner_times_out(monkeypatch):
    async def slow_generate_text(prompt, system_prompt):
        import asyncio

        await asyncio.sleep(1)
        return "{}"

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", slow_generate_text)

    plan = await multimodal_router.plan_request("강의 음성으로 읽어줘", assets=[], planner_timeout_sec=0.01)

    assert plan.steps[0].action == "voice.tts"


def test_rule_fallback_passes_preferred_voice_to_tts_step():
    plan = multimodal_router.fallback_plan(
        "강의 음성으로 읽어줘",
        [],
        quality="high",
        preferred_voice_provider="elevenlabs",
        preferred_voice="voice123",
    )

    assert plan.steps[0].inputs["provider"] == "elevenlabs"
    assert plan.steps[0].inputs["voice"] == "voice123"


def test_rule_fallback_maps_image_analysis():
    asset = MediaAsset(alias="image_1", path="/tmp/a.png", mime_type="image/png")

    plan = multimodal_router.fallback_plan("이 이미지를 분석해줘", [asset], quality="standard")

    assert plan.steps[0].action == "image.analyze"
    assert plan.steps[0].inputs["image"] == "image_1"


def test_rule_fallback_maps_ppt_generation():
    plan = multimodal_router.fallback_plan("AI 윤리 발표자료 만들어줘", [], quality="standard")

    assert plan.steps[0].action == "ppt.generate"
