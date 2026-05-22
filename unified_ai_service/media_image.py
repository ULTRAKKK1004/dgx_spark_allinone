import os
import json
import uuid
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

COMFYUI_URL = "http://localhost:8188"
BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

async def generate_image(prompt: str) -> str:
    """Generates an image via ComfyUI Z-Image Turbo workflow."""
    logger.info(f"Generating image for prompt: {prompt}")
    output_filename = f"image_{uuid.uuid4().hex[:8]}.png"
    
    # Simple Z-Image Turbo JSON (Standard format for ComfyUI)
    # We will use the default text2image workflow structure.
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 12345,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": 1024, "width": 1024}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "text, watermark, low quality, bad anatomy", "clip": ["4", 1]}
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "llm_studio_z", "images": ["8", 0]}
        }
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}) as resp:
                data = await resp.json()
                prompt_id = data.get("prompt_id")
                
            if not prompt_id:
                raise Exception("Failed to queue prompt in ComfyUI")
                
            # Poll for completion
            while True:
                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as h_resp:
                    history = await h_resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        for node_id in outputs:
                            if 'images' in outputs[node_id]:
                                filename = outputs[node_id]['images'][0]['filename']
                                # The file is saved in ComfyUI output dir. We need to copy it.
                                # ComfyUI output dir is mapped to /home/yanus/Docker/output
                                comfy_output_path = f"/home/yanus/Docker/output/{filename}"
                                final_path = os.path.join(RESULTS_DIR, output_filename)
                                import shutil
                                shutil.copy(comfy_output_path, final_path)
                                return final_path
                await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise e
