"""GPU 자원 중재자.

heavy 잡은 vLLM docker stop으로 VRAM 확보 후 실행.
CUDA 안정성을 위해 정지 후 5초 대기.
"""
import asyncio
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar

import aiohttp

logger = logging.getLogger(__name__)

VLLM_CONTAINER = os.getenv("VLLM_CONTAINER", "vllm-server")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8080/v1")
VLLM_RESUME_TIMEOUT = int(os.getenv("VLLM_RESUME_TIMEOUT", "120"))

_lock = asyncio.Lock()
_state = "running"
_heavy_lock_held = ContextVar("heavy_lock_held", default=False)

def vllm_available() -> bool:
    return _state == "running"

def state() -> str:
    return _state

async def _docker(*args: str) -> int:
    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/docker", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode

async def _vllm_healthy() -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            async with session.get(f"{VLLM_URL}/models") as r:
                return r.status == 200
    except:
        return False

async def _pause_vllm() -> bool:
    global _state
    _state = "restarting"
    logger.info("Pausing vLLM...")
    await _docker("stop", VLLM_CONTAINER)
    # Critical: wait for driver cleanup
    await asyncio.sleep(5)
    _state = "paused"
    return True

async def _resume_vllm() -> None:
    global _state
    _state = "restarting"
    logger.info("Resuming vLLM...")
    await _docker("start", VLLM_CONTAINER)
    for _ in range(VLLM_RESUME_TIMEOUT // 2):
        if await _vllm_healthy():
            _state = "running"
            return
        await asyncio.sleep(2)
    _state = "unhealthy"

@asynccontextmanager
async def acquire(vram_class: str):
    if vram_class == "heavy":
        if _heavy_lock_held.get():
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
    else:
        yield
