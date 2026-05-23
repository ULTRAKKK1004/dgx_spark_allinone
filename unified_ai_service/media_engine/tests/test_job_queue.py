import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from media_engine import job_queue


@pytest.fixture
def fake_jm(monkeypatch):
    """job_manager.create_job/update_job 를 in-memory dict로 모의."""
    store = {}

    def create_job(jt, payload):
        jid = f"j-{len(store)+1}"
        store[jid] = {"status": "pending", "type": jt, "input": payload}
        return jid

    def update_job(jid, status, result=None, error=None):
        if jid in store:
            store[jid]["status"] = status
            if result is not None:
                store[jid]["result"] = result
            if error is not None:
                store[jid]["error"] = error

    monkeypatch.setattr(job_queue.job_manager, "create_job", create_job)
    monkeypatch.setattr(job_queue.job_manager, "update_job", update_job)
    return store


@pytest.mark.asyncio
async def test_submit_runs_coro_and_marks_completed(fake_jm, tmp_path):
    target = tmp_path / "x.png"
    target.write_bytes(b"X")

    async def work():
        return target

    jid = await job_queue.submit("test", {"k": "v"}, work())
    for _ in range(100):
        if fake_jm[jid]["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.01)

    assert fake_jm[jid]["status"] == "completed"
    assert fake_jm[jid]["result"].endswith("/api/results/x.png")


@pytest.mark.asyncio
async def test_submit_records_failure(fake_jm):
    async def boom():
        raise RuntimeError("nope")

    jid = await job_queue.submit("test", {}, boom())
    for _ in range(100):
        if fake_jm[jid]["status"] == "failed":
            break
        await asyncio.sleep(0.01)
    assert fake_jm[jid]["status"] == "failed"
    assert "nope" in fake_jm[jid]["error"]


@pytest.mark.asyncio
async def test_submit_text_result(fake_jm):
    """str 반환(예: 비디오 분석 텍스트)은 그대로 결과로."""
    async def work():
        return "this is the analysis"

    jid = await job_queue.submit("test", {}, work())
    for _ in range(100):
        if fake_jm[jid]["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert fake_jm[jid]["result"] == "this is the analysis"
