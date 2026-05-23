import pytest

import main


@pytest.mark.asyncio
async def test_save_multimodal_uploads_assigns_aliases(tmp_path, monkeypatch):
    class FakeUpload:
        def __init__(self, filename, content_type):
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self.filename.encode()

    monkeypatch.setattr(main, "UPLOADS_DIR", str(tmp_path))

    assets = await main._save_multimodal_uploads([FakeUpload("a.png", "image/png")])

    assert assets[0].alias == "image_1"
    assert assets[0].path.endswith("a.png")


@pytest.mark.asyncio
async def test_save_multimodal_uploads_keeps_multiple_files_with_typed_aliases(tmp_path, monkeypatch):
    class FakeUpload:
        def __init__(self, filename, content_type):
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self.filename.encode()

    monkeypatch.setattr(main, "UPLOADS_DIR", str(tmp_path))

    assets = await main._save_multimodal_uploads(
        [
            FakeUpload("first.png", "image/png"),
            FakeUpload("second.jpg", "image/jpeg"),
            FakeUpload("voice.wav", "audio/wav"),
            FakeUpload("clip.mp4", "video/mp4"),
        ]
    )

    assert [asset.alias for asset in assets] == ["image_1", "image_2", "audio_1", "video_1"]
    assert [asset.filename for asset in assets] == ["first.png", "second.jpg", "voice.wav", "clip.mp4"]


@pytest.mark.asyncio
async def test_process_multimodal_task_updates_job(monkeypatch):
    updates = []

    async def fake_plan_request(instruction, assets, quality, preferred_voice_provider, preferred_voice="default"):
        from multimodal_models import MediaPlan

        return MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "text",
                "steps": [{"id": "s1", "action": "text.generate", "inputs": {"prompt": "x"}, "outputs": {"text": "answer"}}],
                "final": {"primary": "answer", "format": "text"},
            },
            supported_actions={"text.generate"},
            asset_aliases=set(),
        )

    async def fake_execute_plan(plan, assets):
        return {"plan": plan.to_dict(), "steps": [], "final": {"type": "text", "value": "ok"}}

    monkeypatch.setattr(main.multimodal_router, "plan_request", fake_plan_request)
    monkeypatch.setattr(main.multimodal_executor, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(main.job_manager, "update_job", lambda *args, **kwargs: updates.append((args, kwargs)))

    await main.process_multimodal_task("job1", "hello", [], "standard", "auto")

    assert updates[0][0] == ("job1", "processing")
    assert updates[-1][0][0] == "job1"
    assert updates[-1][0][1] == "completed"


@pytest.mark.asyncio
async def test_elevenlabs_voices_endpoint_returns_default_and_list(monkeypatch):
    async def fake_list():
        return [{"voice_id": "v1", "name": "Voice One", "category": "premade", "labels": {}}]

    monkeypatch.setattr(main.voice_providers, "list_elevenlabs_voices", fake_list)
    monkeypatch.setattr(main.voice_providers, "get_elevenlabs_voice_id", lambda: "airYK6ydeWdrJg6gyZA3")

    result = await main.elevenlabs_voices_endpoint(auth="test")

    assert result["default_voice_id"] == "airYK6ydeWdrJg6gyZA3"
    assert result["voices"][0]["voice_id"] == "v1"
