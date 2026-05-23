"""이미지 생성·편집 진입점 (media_engine.runner 호출하는 얇은 shim)."""
import logging
import os
import uuid
from pathlib import Path

from media_engine import runner, comfyui_client

logger = logging.getLogger(__name__)


async def generate_image(
    prompt: str,
    workflow: str = "zimage_turbo",
    **kwargs,
) -> Path:
    """텍스트→이미지. workflow ∈ {"zimage_turbo"}."""
    workflow_id = f"image.gen.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, **kwargs)


async def edit_image(
    image_path: str,
    prompt: str,
    workflow: str = "qwen",
    **kwargs,
) -> Path:
    """이미지+프롬프트→편집된 이미지. workflow ∈ {"qwen"}."""
    filename = f"edit_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    await comfyui_client.upload_image(image_path, filename)
    workflow_id = f"image.edit.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, image_name=filename, **kwargs)
