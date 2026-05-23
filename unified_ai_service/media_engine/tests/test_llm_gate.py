import pytest
from unittest.mock import AsyncMock
import sys, os
sys.path.insert(0, "/home/yanus/unified_ai_service")
import llm_service


@pytest.mark.asyncio
async def test_chat_returns_msg_when_vllm_paused(monkeypatch):
    from media_engine import gpu_arbiter
    monkeypatch.setattr(gpu_arbiter, "vllm_available", lambda: False)
    monkeypatch.setattr(gpu_arbiter, "state", lambda: "paused")

    out = await llm_service.generate_text("hi", "sys")
    assert "GPU" in out or "일시" in out or "busy" in out.lower()


@pytest.mark.asyncio
async def test_chat_passes_when_vllm_running(monkeypatch):
    from media_engine import gpu_arbiter
    monkeypatch.setattr(gpu_arbiter, "vllm_available", lambda: True)

    fake_resp = type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "yo"})()})()]})()
    fake_client = type("X", (), {})()
    fake_client.chat = type("Z", (), {})()
    fake_client.chat.completions = type("Q", (), {"create": AsyncMock(return_value=fake_resp)})()
    monkeypatch.setattr(llm_service, "client", fake_client)

    out = await llm_service.generate_text("hi", "sys")
    assert out == "yo"
