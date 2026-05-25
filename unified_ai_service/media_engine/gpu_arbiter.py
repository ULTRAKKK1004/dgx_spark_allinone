"""GPU 자원 중재자.

heavy 잡은 process-wide lock + vLLM docker stop으로 직렬화하고,
종료 시 lock 내에서 docker start + 헬스체크까지 마친 뒤 다음 잡 허용.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar

import aiohttp

logger = logging.getLogger(__name__)

VLLM_CONTAINER = os.getenv("VLLM_CONTAINER", "vllm-server")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1")
VLLM_RESUME_TIMEOUT = int(os.getenv("VLLM_RESUME_TIMEOUT", "420"))

_lock = asyncio.Lock()
_state = "running"  # "running" | "paused" | "restarting" | "unhealthy"
_heavy_lock_held = ContextVar("heavy_lock_held", default=False)


def vllm_available() -> bool:
    return _state == "running"


def state() -> str:
    return _state


async def _docker(*args: str) -> int:
    """docker CLI 호출 — 테스트에서 monkeypatch."""
    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error("docker %s failed (rc=%s): %s", " ".join(args), proc.returncode, stderr.decode())
    else:
        logger.info("docker %s success: %s", " ".join(args), stdout.decode().strip())
    return proc.returncode


async def _vllm_healthy() -> bool:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as session:
            async with session.get(f"{VLLM_URL}/models") as r:
                return r.status == 200
    except Exception:
        return False


async def _pause_vllm() -> None:
    global _state
    _state = "restarting"
    logger.info("Pausing vLLM container for heavy GPU work")
    rc = await _docker("stop", VLLM_CONTAINER)
    if rc != 0:
        logger.warning("docker stop returned %s — continuing anyway", rc)
    
    # Wait for VRAM to be freed (up to 15 seconds)
    import subprocess
    import time
    for _ in range(15):
        try:
            # Check if any process with 'vllm' in its name is still using GPU
            # We look for processes in nvidia-smi output
            out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=name", "--format=csv,noheader"], text=True)
            if "vllm" not in out.lower() and "VLLM" not in out:
                logger.info("vLLM memory freed successfully")
                break
        except Exception:
            break
        await asyncio.sleep(1)
    
    _state = "paused"


async def _resume_vllm() -> None:
    global _state
    _state = "restarting"
    logger.info("Resuming vLLM container")
    await _docker("start", VLLM_CONTAINER)
    for _ in range(VLLM_RESUME_TIMEOUT):
        if await _vllm_healthy():
            _state = "running"
            logger.info("vLLM is healthy again")
            return
        await asyncio.sleep(1)
    _state = "unhealthy"
    logger.error("vLLM failed to become healthy within %ds", VLLM_RESUME_TIMEOUT)


@asynccontextmanager
async def acquire(vram_class: str):
    """heavy: 직렬화 + vLLM swap (재진입 가능). light: pass-through."""
    if vram_class == "heavy":
        if _heavy_lock_held.get():
            # 이미 이 태스크에서 heavy 락을 쥐고 있음 (재진입)
            yield
            return

        async with _lock:
            token = _heavy_lock_held.set(True)
            try:
                await _pause_vllm()
                try:
                    yield
                finally:
                    await _resume_vllm()
            finally:
                _heavy_lock_held.reset(token)
    elif vram_class == "light":
        yield
    else:
        raise ValueError(f"unknown vram_class: {vram_class!r}")
