"""이미지 생성·편집·변환 진입점 (media_engine.runner 호출하는 얇은 shim)."""
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
    """텍스트→이미지. workflow in {"zimage_turbo", "flux"}."""
    # Ensure explicit args are not in kwargs
    kwargs.pop("prompt", None)
    kwargs.pop("workflow", None)
    workflow_id = f"image.gen.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, **kwargs)


async def edit_image(
    image_path: str,
    prompt: str,
    workflow: str = "qwen",
    **kwargs,
) -> Path:
    """이미지+프롬프트→편집된 이미지. workflow in {"qwen"}."""
    kwargs.pop("prompt", None)
    kwargs.pop("image_name", None)
    filename = f"edit_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    await comfyui_client.upload_image(image_path, filename)
    workflow_id = f"image.edit.{workflow}"
    return await runner.run(workflow_id, prompt=prompt, image_name=filename, **kwargs)


async def control_image(
    prompt: str,
    control_image_path: str,
    control_type: str = "canny",
    strength: float = 0.7,
    **kwargs,
) -> Path:
    """FLUX ControlNet으로 구도/자세/깊이 제어 이미지 생성."""
    filename = f"ctrl_{uuid.uuid4().hex[:8]}_{os.path.basename(control_image_path)}"
    await comfyui_client.upload_image(control_image_path, filename)
    return await runner.run(
        "image.ctrl.flux_union",
        prompt=prompt,
        control_image=filename,
        control_type=control_type,
        strength=strength,
        **kwargs,
    )


async def inpaint_image(
    image_path: str,
    mask_path: str,
    prompt: str,
    **kwargs,
) -> Path:
    """Qwen Image Edit으로 마스크 영역만 instruction-edit."""
    img_filename = f"inp_img_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    msk_filename = f"inp_msk_{uuid.uuid4().hex[:8]}_{os.path.basename(mask_path)}"
    await comfyui_client.upload_image(image_path, img_filename)
    await comfyui_client.upload_image(mask_path, msk_filename)
    return await runner.run(
        "image.inpaint.qwen",
        prompt=prompt,
        image_name=img_filename,
        mask_name=msk_filename,
        **kwargs,
    )


async def analyze_image_janus(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    **kwargs,
) -> str:
    """Janus-Pro 7B를 이용한 고성능 이미지 분석."""
    filename = f"janus_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
    await comfyui_client.upload_image(image_path, filename)
    res = await runner.run(
        "image.analyze.janus",
        image_name=filename,
        prompt=prompt,
        **kwargs,
    )
    return str(res)
