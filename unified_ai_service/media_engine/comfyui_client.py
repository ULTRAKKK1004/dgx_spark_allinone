"""ComfyUI HTTP API 클라이언트 — submit / poll / upload / fetch."""
import asyncio
import os
from pathlib import Path

import aiohttp

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://localhost:8188")
COMFY_OUTPUT_DIR = os.getenv("COMFYUI_OUTPUT_DIR", "/home/yanus/Docker/output")


class ComfyUIError(RuntimeError):
    pass


async def submit(workflow_prompt: dict) -> str:
    """워크플로우를 큐에 등록하고 prompt_id 반환."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow_prompt},
        ) as resp:
            data = await resp.json()
            if resp.status != 200 or "prompt_id" not in data:
                node_errs = data.get("node_errors") or data.get("error") or data
                raise ComfyUIError(f"submit rejected: {node_errs}")
            return data["prompt_id"]


async def _get_history(prompt_id: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
            return await resp.json()


async def wait_and_fetch(
    prompt_id: str,
    output_node: str,
    timeout: float = 300,
    poll_interval: float = 2.0,
) -> Path:
    """ComfyUI 워크플로우 완료를 기다리고 출력 파일 경로를 반환."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        history = await _get_history(prompt_id)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                msg = next(
                    (m[1].get("exception_message") for m in msgs if m and m[0] == "error"),
                    str(status),
                )
                raise ComfyUIError(f"workflow error: {msg}")

            outputs = entry.get("outputs", {})
            node_out = outputs.get(output_node)
            if node_out:
                filename = _extract_filename(node_out)
                if filename:
                    return Path(COMFY_OUTPUT_DIR) / filename
        await asyncio.sleep(poll_interval)
    raise ComfyUIError(f"timeout after {timeout}s for prompt {prompt_id}")


def _extract_filename(node_output: dict) -> str | None:
    """SaveImage(images) / VHS_VideoCombine(gifs|videos) 양쪽 모두 대응."""
    for key in ("images", "gifs", "videos"):
        items = node_output.get(key)
        if items:
            return items[0]["filename"]
    return None


async def upload_image(local_path: str, filename: str) -> dict:
    """ComfyUI input/ 디렉토리로 이미지 업로드."""
    async with aiohttp.ClientSession() as session:
        with open(local_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("image", f, filename=filename)
            form.add_field("type", "input")
            async with session.post(f"{COMFYUI_URL}/upload/image", data=form) as resp:
                data = await resp.json()
                if resp.status != 200:
                    raise ComfyUIError(f"upload failed: {data}")
                return data


async def upload_audio(local_path: str, filename: str) -> dict:
    """ComfyUI input/ 디렉토리로 오디오 업로드 (image 엔드포인트 재사용)."""
    return await upload_image(local_path, filename)
