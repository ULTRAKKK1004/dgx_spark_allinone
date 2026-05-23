import pytest

import main


@pytest.mark.asyncio
async def test_save_multimodal_uploads_assigns_aliases(tmp_path, monkeypatch):
    class FakeUpload:
        filename = "a.png"
        content_type = "image/png"

        async def read(self):
            return b"png"

    monkeypatch.setattr(main, "UPLOADS_DIR", str(tmp_path))

    assets = await main._save_multimodal_uploads([FakeUpload()])

    assert assets[0].alias == "image_1"
    assert assets[0].path.endswith("a.png")


@pytest.mark.asyncio
async def test_process_multimodal_task_updates_job(monkeypatch):
    updates = []

    async def fake_plan_request(instruction, assets, quality, preferred_voice_provider):
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
