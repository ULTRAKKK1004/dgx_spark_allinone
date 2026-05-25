"""media_engine 실행 진입점 — render → submit → poll → fetch → copy."""
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from media_engine import catalog, comfyui_client, gpu_arbiter

logger = logging.getLogger(__name__)

RESULTS_DIR = "/home/yanus/unified_ai_service/results"
WF_DIR = Path(__file__).parent / "workflows"

_env = Environment(
    loader=FileSystemLoader(str(WF_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    keep_trailing_newline=False,
)
_env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)


def render_template(template: str, params: dict) -> dict:
    rendered = _env.get_template(template).render(**params)
    return json.loads(rendered)


async def run(workflow_id: str, **params) -> Path | str:
    meta = catalog.get(workflow_id)
    validated = catalog.validate(meta, params)
    workflow = render_template(meta["template"], validated)

    logger.info("Running workflow %s (vram_class=%s)", workflow_id, meta["vram_class"])

    async with gpu_arbiter.acquire(meta["vram_class"]):
        prompt_id = await comfyui_client.submit(workflow)
        res = await comfyui_client.wait_and_fetch(
            prompt_id,
            output_node=meta["output_node"],
            timeout=meta["timeout_sec"],
        )

    if isinstance(res, str):
        logger.info("Workflow %s done (string result)", workflow_id)
        return res

    # Proceed with Path handling
    source = res
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = source.suffix or ".bin"
    dest = Path(RESULTS_DIR) / f"{workflow_id.replace('.', '_')}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy(source, dest)
    logger.info("Workflow %s done: %s", workflow_id, dest)
    return dest
