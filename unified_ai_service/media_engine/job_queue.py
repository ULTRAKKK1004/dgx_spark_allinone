"""기존 job_manager 위 직렬화 래퍼.

heavy 잡의 GPU 직렬화는 gpu_arbiter의 lock이 담당하므로 여기서는 단순 등록·실행만.
실패 시 에러 로그를 results/_errors/ 에 덤프한다.
"""
import asyncio
import logging
import os
import traceback
from pathlib import Path

import sys
sys.path.insert(0, "/home/yanus/unified_ai_service")
import job_manager  # unified_ai_service 의 기존 모듈

logger = logging.getLogger(__name__)

ERROR_DIR = "/home/yanus/unified_ai_service/results/_errors"


async def submit(job_type: str, payload: dict, coro, user_email: str = "Guest") -> str:
    """잡을 등록하고 background에서 coro 실행."""
    job_id = job_manager.create_job(job_type, payload, user_email=user_email)
    asyncio.create_task(_run(job_id, coro))
    return job_id


async def _run(job_id: str, coro) -> None:
    try:
        job_manager.update_job(job_id, "processing")
        result = await coro
        if isinstance(result, (Path, os.PathLike)):
            filename = os.path.basename(str(result))
            job_manager.update_job(job_id, "completed", result=f"/api/results/{filename}")
        else:
            job_manager.update_job(job_id, "completed", result=result)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Job %s failed: %s", job_id, tb)
        _dump_error(job_id, tb)
        job_manager.update_job(job_id, "failed", error=str(e))


def _dump_error(job_id: str, traceback_text: str) -> None:
    try:
        os.makedirs(ERROR_DIR, exist_ok=True)
        with open(os.path.join(ERROR_DIR, f"{job_id}.log"), "w") as f:
            f.write(traceback_text)
    except Exception:
        pass
