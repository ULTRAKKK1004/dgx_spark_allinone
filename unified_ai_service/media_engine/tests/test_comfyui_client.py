"""ComfyUI HTTP 클라이언트 동작 검증 (aiohttp 응답 mock)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from media_engine import comfyui_client as cc


@pytest.mark.asyncio
async def test_submit_returns_prompt_id():
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.json = AsyncMock(return_value={"prompt_id": "abc-123"})
    fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_resp.__aexit__ = AsyncMock(return_value=False)

    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=fake_resp)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    with patch.object(cc.aiohttp, "ClientSession", return_value=fake_session):
        pid = await cc.submit({"3": {"class_type": "X"}})
        assert pid == "abc-123"


@pytest.mark.asyncio
async def test_submit_raises_on_reject():
    fake_resp = MagicMock()
    fake_resp.status = 400
    fake_resp.json = AsyncMock(return_value={"error": "node_errors", "node_errors": {"3": "boom"}})
    fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
    fake_resp.__aexit__ = AsyncMock(return_value=False)

    fake_session = MagicMock()
    fake_session.post = MagicMock(return_value=fake_resp)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    with patch.object(cc.aiohttp, "ClientSession", return_value=fake_session):
        with pytest.raises(cc.ComfyUIError, match="boom"):
            await cc.submit({"3": {"class_type": "X"}})


@pytest.mark.asyncio
async def test_wait_and_fetch_polls_until_done(tmp_path, monkeypatch):
    """history가 비어있다가 → 채워지면 출력 파일 경로 반환."""
    monkeypatch.setattr(cc, "COMFY_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "result.png").write_bytes(b"PNG")

    history_states = [
        {},  # 첫 poll: 미완료
        {"abc-123": {"outputs": {"9": {"images": [{"filename": "result.png", "type": "output"}]}}}},
    ]

    async def fake_get_history(prompt_id):
        return history_states.pop(0) if history_states else history_states[-1]

    monkeypatch.setattr(cc, "_get_history", fake_get_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    path = await cc.wait_and_fetch("abc-123", output_node="9", timeout=10)
    assert str(path).endswith("result.png")


@pytest.mark.asyncio
async def test_wait_and_fetch_timeout(monkeypatch):
    async def empty_history(_):
        return {}
    monkeypatch.setattr(cc, "_get_history", empty_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    with pytest.raises(cc.ComfyUIError, match="timeout"):
        await cc.wait_and_fetch("abc-123", output_node="9", timeout=1, poll_interval=0.01)


@pytest.mark.asyncio
async def test_wait_and_fetch_workflow_error(monkeypatch):
    """history 응답에 status.error가 있으면 ComfyUIError로 전환."""
    async def err_history(_):
        return {"abc-123": {"status": {"status_str": "error", "messages": [["error", {"exception_message": "OOM"}]]}}}
    monkeypatch.setattr(cc, "_get_history", err_history)
    monkeypatch.setattr(cc.asyncio, "sleep", AsyncMock())

    with pytest.raises(cc.ComfyUIError, match="OOM"):
        await cc.wait_and_fetch("abc-123", output_node="9", timeout=5, poll_interval=0.01)
