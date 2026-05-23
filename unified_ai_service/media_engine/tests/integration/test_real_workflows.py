"""실제 GPU + ComfyUI + vLLM 통합 시험.

실행: ./venv/bin/python -m pytest media_engine/tests/integration -v --integration -s --timeout=1800
"""
import asyncio
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, "/home/yanus/unified_ai_service")

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_zimage_turbo_end_to_end():
    """light 워크플로우: vLLM swap 없이 실행되어야 한다."""
    from media_engine import runner, gpu_arbiter
    initial = gpu_arbiter.state()
    out = await runner.run(
        "image.gen.zimage_turbo",
        prompt="a small cat sitting in a sunlit room",
        steps=4,
    )
    assert out.exists(), f"output not produced: {out}"
    assert out.stat().st_size > 1000
    assert gpu_arbiter.state() == initial, "light job must not change vLLM state"


@pytest.mark.asyncio
async def test_wan22_i2v_vllm_swap_cycle(tmp_path):
    """heavy 워크플로우: 진입 시 vLLM stop, 종료 후 자동 start."""
    from media_engine import runner, gpu_arbiter, comfyui_client

    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(120, 80, 200))
    seed_path = tmp_path / "seed.png"
    img.save(seed_path)
    await comfyui_client.upload_image(str(seed_path), "integ_seed.png")

    assert gpu_arbiter.vllm_available(), "precondition: vLLM should be running"

    out = await runner.run(
        "video.i2v.wan22",
        prompt="gentle camera motion, cinematic",
        image_name="integ_seed.png",
        frames=33,  # ~2s for speed
        steps=4,
    )
    assert out.exists() and out.stat().st_size > 10_000

    # vLLM 자동 복구 확인 (최대 120초)
    for _ in range(120):
        if gpu_arbiter.vllm_available():
            break
        await asyncio.sleep(1)
    assert gpu_arbiter.vllm_available(), f"vLLM did not recover, state={gpu_arbiter.state()}"


@pytest.mark.asyncio
async def test_serialized_two_heavy_jobs():
    """동시 heavy 잡 2건이 순차 처리되며 둘 다 성공."""
    from media_engine import runner, comfyui_client
    from PIL import Image
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        img = Image.new("RGB", (512, 512), color=(50, 200, 100))
        p = Path(td) / "s.png"
        img.save(p)
        await comfyui_client.upload_image(str(p), "integ_s.png")

        async def job(idx):
            return await runner.run(
                "video.i2v.wan22",
                prompt=f"motion #{idx}",
                image_name="integ_s.png",
                frames=33,
                steps=4,
            )

        out1, out2 = await asyncio.gather(job(1), job(2))
        assert out1.exists() and out2.exists()
        assert out1 != out2
