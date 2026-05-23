import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from media_engine import runner, catalog


@pytest.mark.asyncio
async def test_run_renders_validates_submits_fetches(tmp_path, monkeypatch):
    """runner.run은 catalog 검증 + 템플릿 렌더 + ComfyUI submit + fetch를 엮는다."""
    fake_submit = AsyncMock(return_value="pid-1")
    fake_fetch = AsyncMock(return_value=tmp_path / "out.png")
    (tmp_path / "out.png").write_bytes(b"PNG")

    monkeypatch.setattr(runner.comfyui_client, "submit", fake_submit)
    monkeypatch.setattr(runner.comfyui_client, "wait_and_fetch", fake_fetch)
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    out_path = await runner.run("image.gen.zimage_turbo", prompt="hello world")

    assert out_path.exists()
    assert out_path.parent.name == "results"
    wf = fake_submit.call_args[0][0]
    flat = json.dumps(wf, ensure_ascii=False)
    assert "hello world" in flat


@pytest.mark.asyncio
async def test_run_uses_gpu_arbiter(monkeypatch, tmp_path):
    """heavy 워크플로우는 gpu_arbiter.acquire('heavy')를 통해 실행된다."""
    captured = []

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def fake_acquire(vc):
        captured.append(vc)
        yield

    monkeypatch.setattr(runner.gpu_arbiter, "acquire", fake_acquire)
    monkeypatch.setattr(runner.comfyui_client, "submit", AsyncMock(return_value="pid"))
    (tmp_path / "x.mp4").write_bytes(b"MP4")
    monkeypatch.setattr(runner.comfyui_client, "wait_and_fetch", AsyncMock(return_value=tmp_path / "x.mp4"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    await runner.run("video.i2v.wan22", prompt="p", image_name="a.png")
    assert captured == ["heavy"]


@pytest.mark.asyncio
async def test_run_missing_param_raises():
    with pytest.raises(ValueError, match="prompt"):
        await runner.run("image.gen.zimage_turbo")


@pytest.mark.asyncio
async def test_run_unknown_workflow_raises():
    with pytest.raises(KeyError, match="unknown"):
        await runner.run("not.real.workflow", prompt="x")
