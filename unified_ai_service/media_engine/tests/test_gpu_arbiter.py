import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from media_engine import gpu_arbiter as ga


@pytest.fixture(autouse=True)
def _reset_state():
    ga._state = "running"
    yield
    ga._state = "running"


@pytest.mark.asyncio
async def test_light_does_not_lock(monkeypatch):
    """light 작업은 lock 없이 통과하고 vLLM은 그대로 running."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))

    async with ga.acquire("light"):
        pass

    assert ga.vllm_available() is True
    fake_run.assert_not_called()


@pytest.mark.asyncio
async def test_heavy_pauses_and_resumes_vllm(monkeypatch):
    """heavy 작업은 docker stop → 작업 → docker start 트리거."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())

    async with ga.acquire("heavy"):
        assert any("stop" in str(c) for c in fake_run.call_args_list), \
            f"expected docker stop, got {fake_run.call_args_list}"
        assert ga.vllm_available() is False

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    stop_calls = [c for c in fake_run.call_args_list if "stop" in str(c)]
    start_calls = [c for c in fake_run.call_args_list if "start" in str(c)]
    assert len(stop_calls) >= 1
    for _ in range(50):
        if start_calls:
            break
        await asyncio.sleep(0.01)
        start_calls = [c for c in fake_run.call_args_list if "start" in str(c)]
    assert len(start_calls) >= 1


@pytest.mark.asyncio
async def test_heavy_serializes(monkeypatch):
    """동시 heavy 잡 2건은 순차 실행."""
    fake_run = AsyncMock(return_value=0)
    monkeypatch.setattr(ga, "_docker", fake_run)
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=True))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())

    order = []

    async def heavy_work(name):
        async with ga.acquire("heavy"):
            order.append(f"{name}-enter")
            await asyncio.sleep(0)
            order.append(f"{name}-leave")

    await asyncio.gather(heavy_work("A"), heavy_work("B"))

    assert order in (
        ["A-enter", "A-leave", "B-enter", "B-leave"],
        ["B-enter", "B-leave", "A-enter", "A-leave"],
    )


@pytest.mark.asyncio
async def test_vllm_resume_failure_marks_unhealthy(monkeypatch):
    """healthcheck 실패 시 vllm_state=unhealthy."""
    monkeypatch.setattr(ga, "_docker", AsyncMock(return_value=0))
    monkeypatch.setattr(ga, "_vllm_healthy", AsyncMock(return_value=False))
    monkeypatch.setattr(ga.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(ga, "VLLM_RESUME_TIMEOUT", 3)

    async with ga.acquire("heavy"):
        pass

    for _ in range(100):
        if ga._state == "unhealthy":
            break
        await asyncio.sleep(0.01)
    assert ga._state == "unhealthy"
    assert ga.vllm_available() is False
