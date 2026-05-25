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


def test_rule_fallback_builds_voiced_lecture_video_from_image_and_script_file():
    image = MediaAsset(alias="image_1", path="/tmp/lecturer.png", mime_type="image/png")
    script = MediaAsset(
        alias="file_1",
        path="/tmp/script.md",
        mime_type="text/markdown",
        filename="script.md",
    )

    plan = multimodal_router.fallback_plan(
        "입력한 이미지가 강의장에 서있는 이미지입니다. 입력한 강의 대본대로 강의하는 영상을 만들어주세요.",
        [image, script],
        quality="high",
        preferred_voice_provider="elevenlabs",
        preferred_voice="voice123",
    )

    assert [step.action for step in plan.steps] == [
        "text.extract",
        "voice.tts",
        "video.lecture",
    ]
    assert plan.steps[0].inputs == {"file": "file_1"}
    assert plan.steps[1].inputs["text"] == "$script_text"
    assert plan.steps[1].inputs["provider"] == "elevenlabs"
    assert plan.steps[1].inputs["voice"] == "voice123"
    assert plan.steps[2].inputs["image"] == "image_1"
    assert plan.steps[2].inputs["audio"] == "$voice_audio"
    assert plan.final == {"primary": "lecture_video", "format": "video"}


def test_rule_fallback_uses_audio_conditioned_talking_video_for_lecture():
    image = MediaAsset(alias="image_1", path="/tmp/lecturer.png", mime_type="image/png")
    script = MediaAsset(alias="file_1", path="/tmp/script.md", mime_type="text/markdown")

    plan = multimodal_router.fallback_plan(
        "강사의 입모양도 음성과 자연스럽게 싱크가 맞고 조금씩 움직이면서 강의하게 해줘",
        [image, script],
        quality="high",
        preferred_voice_provider="elevenlabs",
        preferred_voice="voice123",
    )

    assert [step.action for step in plan.steps] == [
        "text.extract",
        "voice.tts",
        "video.lecture",
    ]
    assert plan.steps[2].inputs["image"] == "image_1"
    assert plan.steps[2].inputs["audio"] == "$voice_audio"
    assert plan.steps[2].outputs == {"video": "lecture_video"}


def test_lecture_video_keeps_auto_provider_even_when_voice_id_is_selected():
    image = MediaAsset(alias="image_1", path="/tmp/lecturer.png", mime_type="image/png")
    script = MediaAsset(alias="file_1", path="/tmp/script.md", mime_type="text/markdown")

    plan = multimodal_router.fallback_plan(
        "강의 대본대로 입모양 싱크 맞는 영상을 만들어줘",
        [image, script],
        quality="standard",
        preferred_voice_provider="auto",
        preferred_voice="airYK6ydeWdrJg6gyZA3",
    )

    assert plan.steps[1].inputs["provider"] == "auto"
    assert plan.steps[1].inputs["voice"] == "airYK6ydeWdrJg6gyZA3"


@pytest.mark.asyncio
async def test_plan_request_forces_voiced_lecture_video_even_if_planner_would_skip_script(monkeypatch):
    image = MediaAsset(alias="image_1", path="/tmp/lecturer.png", mime_type="image/png")
    script = MediaAsset(alias="file_1", path="/tmp/script.txt", mime_type="text/plain")

    async def planner_should_not_run(prompt, system_prompt):
        raise AssertionError("lecture script routing should not depend on the LLM planner")

    monkeypatch.setattr(multimodal_router.llm_service, "generate_text", planner_should_not_run)

    plan = await multimodal_router.plan_request(
        "대본대로 자연스럽게 강의하는 영상을 만들어줘",
        [image, script],
    )

    assert [step.action for step in plan.steps] == [
        "text.extract",
        "voice.tts",
        "video.lecture",
    ]
