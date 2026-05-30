import os
import uuid
import base64
import asyncio
import logging
import shutil
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request, BackgroundTasks

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import llm_service
import stt_service
import ppt_service
import auth_service
import job_manager

import media_audio
import media_video
import media_image
import multimodal_router
import multimodal_executor
import voice_providers

from media_engine import job_queue, gpu_arbiter
from multimodal_models import MediaAsset


app = FastAPI(title="AI Hub")
templates = Jinja2Templates(directory="templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tor-ai.com",
        "https://www.tor-ai.com",
        "https://tube.tor-ai.com",
        "https://vllmapi.tor-ai.com",
        "https://comfyui.tor-ai.com",
        "https://blog.tor-ai.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "/home/yanus/unified_ai_service"
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

async def flexible_auth(request: Request, authorization: Optional[str] = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "")
        if auth_service.verify_token(token):
            return {"type": "token", "token": token, "email": "admin@tor-ai.com"} # Treat valid token as admin
        logger.warning(f"Invalid API Token: {token[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API Token")

    referer = request.headers.get("referer", "")
    host = request.headers.get("host", "")
    logger.info(f"Auth Check - Host: {host}, Referer: {referer}")

    if (host and host in referer) or not referer:
        return {"type": "web_ui", "email": request.headers.get("X-Email", "Guest")}

    logger.warning(f"Auth Failed - Host: {host}, Referer: {referer}")
    raise HTTPException(status_code=401, detail="Missing Authorization Header or Invalid Referer")
class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful AI assistant running on DGX Spark."
    history: List[Dict[str, str]] = [] # For chat history

# Background Task Workers
async def process_llm_task(job_id: str, req: ChatRequest):
    try:
        logger.info(f"Starting LLM Task: {job_id}")
        job_manager.update_job(job_id, "processing")
        
        # Build prompt with history
        messages = [{"role": "system", "content": req.system_prompt}]
        for msg in req.history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": req.prompt})
        
        full_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages[1:]])
        logger.info(f"LLM Prompt built, length: {len(full_prompt)}")
        
        response_text = await llm_service.generate_text(full_prompt, req.system_prompt)
        job_manager.update_job(job_id, "completed", result=response_text)
        logger.info(f"LLM Task Completed: {job_id}")
    except Exception as e:
        logger.error(f"LLM Task Failed: {job_id}, Error: {e}")
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_vlm_task(job_id: str, data_url: str, prompt: str):
    try:
        job_manager.update_job(job_id, "processing")
        response_text = await llm_service.analyze_image(data_url, prompt)
        job_manager.update_job(job_id, "completed", result=response_text)
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_stt_task(job_id: str, filepath: str):
    try:
        job_manager.update_job(job_id, "processing")
        text = await stt_service.transcribe_audio(filepath)
        job_manager.update_job(job_id, "completed", result=text)
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_ppt_task(job_id: str, topic: str):
    try:
        job_manager.update_job(job_id, "processing")
        slides_data = await llm_service.generate_ppt_structure(topic)
        output_filename = f"presentation_{job_id}.pptx"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        ppt_service.generate_ppt_file(title=topic, slides_data=slides_data, output_path=output_path)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{output_filename}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))


def _asset_alias_for_upload(index: int, upload: UploadFile) -> str:
    mime = upload.content_type or "application/octet-stream"
    if mime.startswith("image/"):
        prefix = "image"
    elif mime.startswith("audio/"):
        prefix = "audio"
    elif mime.startswith("video/"):
        prefix = "video"
    else:
        prefix = "file"
    return f"{prefix}_{index}"


async def _save_multimodal_uploads(files: Optional[List[UploadFile]]) -> List[MediaAsset]:
    assets: List[MediaAsset] = []
    if not files:
        return assets
    counters = {"image": 0, "audio": 0, "video": 0, "file": 0}
    for upload in files:
        mime = upload.content_type or "application/octet-stream"
        if mime.startswith("image/"):
            kind = "image"
        elif mime.startswith("audio/"):
            kind = "audio"
        elif mime.startswith("video/"):
            kind = "video"
        else:
            kind = "file"
        counters[kind] += 1
        alias = f"{kind}_{counters[kind]}"
        safe_name = os.path.basename(upload.filename or f"{alias}.bin")
        path = os.path.join(UPLOADS_DIR, f"multi_{uuid.uuid4().hex}_{safe_name}")
        with open(path, "wb") as f:
            f.write(await upload.read())
        assets.append(MediaAsset(alias=alias, path=path, mime_type=mime, filename=safe_name))
    return assets


async def process_multimodal_task(
    job_id: str,
    instruction: str,
    assets: List[MediaAsset],
    quality: str,
    preferred_voice_provider: str,
    preferred_voice: str = "default",
):
    try:
        job_manager.update_job(job_id, "processing")
        plan = await multimodal_router.plan_request(
            instruction,
            assets,
            quality=quality,
            preferred_voice_provider=preferred_voice_provider,
            preferred_voice=preferred_voice,
        )
        result = await multimodal_executor.execute_plan(plan, assets)
        job_manager.update_job(job_id, "completed", result=result)
    except Exception as e:
        logger.error("Multimodal job failed: %s", e, exc_info=True)
        job_manager.update_job(job_id, "failed", error=str(e))


# Endpoints

@app.post("/api/multimodal/execute")
async def multimodal_execute_endpoint(
    background_tasks: BackgroundTasks,
    instruction: str = Form(...),
    quality: str = Form("standard"),
    preferred_voice_provider: str = Form("auto"),
    preferred_voice: str = Form("default"),
    files: Optional[List[UploadFile]] = File(None),
    dry_run: bool = Form(False),
    auth = Depends(flexible_auth),
):
    if quality not in {"draft", "standard", "high"}:
        raise HTTPException(status_code=400, detail="quality must be draft, standard, or high")
    if preferred_voice_provider not in {"auto", "local_f5", "elevenlabs"}:
        raise HTTPException(status_code=400, detail="preferred_voice_provider must be auto, local_f5, or elevenlabs")
    
    assets = await _save_multimodal_uploads(files)
    
    if dry_run:
        plan = await multimodal_router.plan_request(
            instruction, assets, quality, preferred_voice_provider, preferred_voice
        )
        return {"dry_run": True, "plan": plan.to_dict(), "assets": [a.to_dict() for a in assets]}

    # Deduct quota (Phase C)
    user_email = auth.get("email")
    if user_email and user_email not in ["Guest", "yeonwoo.kim03@gmail.com", "admin@tor-ai.com"]:
        async with aiohttp.ClientSession() as session:
            # 10 credits per multimodal task
            async with session.post(
                "http://localhost:8002/api/user/quota/deduct",
                params={"email": user_email, "type": "credits", "amount": 10}
            ) as resp:
                if resp.status == 403:
                    raise HTTPException(status_code=403, detail="Insufficient credits")

    job_id = job_manager.create_job(
        "multimodal",
        {
            "instruction": instruction,
            "quality": quality,
            "preferred_voice_provider": preferred_voice_provider,
            "preferred_voice": preferred_voice,
            "assets": [asset.to_dict() for asset in assets],
        },
        user_email=user_email
    )
    background_tasks.add_task(process_multimodal_task, job_id, instruction, assets, quality, preferred_voice_provider, preferred_voice)
    return {"job_id": job_id}


@app.get("/api/elevenlabs/voices")
async def elevenlabs_voices_endpoint(auth = Depends(flexible_auth)):
    voices = await voice_providers.list_elevenlabs_voices()
    return {
        "default_voice_id": voice_providers.get_elevenlabs_voice_id(),
        "configured": bool(os.getenv("ELEVENLABS_API_KEY")),
        "voices": voices,
    }


@app.post("/api/media/image")
async def generate_image_endpoint(prompt: str = Form(...), workflow: str = Form("zimage_turbo"), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    coro = media_image.generate_image(prompt, workflow=workflow)
    job_id = await job_queue.submit("media_image", {"prompt": prompt, "workflow": workflow}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/image/edit")
async def edit_image_endpoint(prompt: str = Form(...), image: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    img_path = os.path.join(UPLOADS_DIR, f"edit_in_{uuid.uuid4().hex}_{image.filename}")
    with open(img_path, "wb") as f:
        f.write(await image.read())
    coro = media_image.edit_image(img_path, prompt)
    job_id = await job_queue.submit("media_image_edit", {"prompt": prompt}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/image/control")
async def control_image_endpoint(
    prompt: str = Form(...),
    control_type: str = Form("canny"),
    strength: float = Form(0.7),
    control_image: UploadFile = File(...),
    auth = Depends(flexible_auth),
):
    user_email = auth.get("email")
    img_path = os.path.join(UPLOADS_DIR, f"ctrl_in_{uuid.uuid4().hex}_{control_image.filename}")
    with open(img_path, "wb") as f:
        f.write(await control_image.read())
    coro = media_image.control_image(prompt, img_path, control_type=control_type, strength=strength)
    job_id = await job_queue.submit(
        "media_image_control",
        {"prompt": prompt, "control_type": control_type, "strength": strength},
        coro,
        user_email=user_email
    )
    return {"job_id": job_id}

@app.post("/api/media/image/inpaint")
async def inpaint_image_endpoint(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    auth = Depends(flexible_auth),
):
    user_email = auth.get("email")
    img_path = os.path.join(UPLOADS_DIR, f"inp_img_{uuid.uuid4().hex}_{image.filename}")
    msk_path = os.path.join(UPLOADS_DIR, f"inp_msk_{uuid.uuid4().hex}_{mask.filename}")
    with open(img_path, "wb") as f:
        f.write(await image.read())
    with open(msk_path, "wb") as f:
        f.write(await mask.read())
    coro = media_image.inpaint_image(img_path, msk_path, prompt)
    job_id = await job_queue.submit("media_image_inpaint", {"prompt": prompt}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/music")
async def generate_music_endpoint(prompt: str = Form(...), duration: int = Form(10), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    if duration > 30:
        coro = media_audio.generate_long_music(prompt, duration)
    else:
        coro = media_audio.generate_music(prompt, duration)
    job_id = await job_queue.submit("media_music", {"prompt": prompt, "duration": duration}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/tts")
async def generate_tts_endpoint(text: str = Form(...), ref_audio: UploadFile = File(None), ref_text: str = Form(""), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    ref_path = ""
    if ref_audio:
        ref_path = os.path.join(UPLOADS_DIR, f"ref_{uuid.uuid4().hex}_{ref_audio.filename}")
        with open(ref_path, "wb") as f:
            f.write(await ref_audio.read())
    coro = media_audio.generate_tts_with_effects(text, ref_path, ref_text)
    job_id = await job_queue.submit("media_tts", {"text": text}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/video/gen")
async def generate_video_endpoint(prompt: str = Form(...), duration: int = Form(30), base_image: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    img_path = os.path.join(UPLOADS_DIR, f"base_img_{uuid.uuid4().hex}_{base_image.filename}")
    with open(img_path, "wb") as f:
        f.write(await base_image.read())
    coro = media_video.generate_long_video(prompt, img_path, duration)
    job_id = await job_queue.submit("media_video_gen", {"prompt": prompt, "duration": duration}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/video/edit")
async def edit_video_endpoint(prompt: str = Form(""), video: UploadFile = File(...), audio: UploadFile = File(None), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    vid_path = os.path.join(UPLOADS_DIR, f"edit_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    aud_path = ""
    if audio:
        aud_path = os.path.join(UPLOADS_DIR, f"edit_aud_{uuid.uuid4().hex}_{audio.filename}")
        with open(aud_path, "wb") as f:
            f.write(await audio.read())
    coro = media_video.edit_video(vid_path, aud_path, prompt=prompt)
    job_id = await job_queue.submit("media_video_edit", {"prompt": prompt}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/video/shorts")
async def video_shorts_endpoint(prompt: str = Form(""), video: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    vid_path = os.path.join(UPLOADS_DIR, f"shorts_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    coro = media_video.shorten_video(vid_path, prompt)
    job_id = await job_queue.submit("media_video_shorts", {"prompt": prompt}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/media/video/analyze")
async def video_analyze_endpoint(prompt: str = Form(""), video: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    vid_path = os.path.join(UPLOADS_DIR, f"analyze_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
    coro = media_video.analyze_video(vid_path, prompt)
    job_id = await job_queue.submit("media_video_analyze", {"prompt": prompt}, coro, user_email=user_email)
    return {"job_id": job_id}

@app.post("/api/llm/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks, auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    job_id = job_manager.create_job("llm", {"prompt": req.prompt}, user_email=user_email)
    background_tasks.add_task(process_llm_task, job_id, req)
    return {"job_id": job_id}

@app.post("/api/vlm/analyze")
async def analyze_image_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(...), image: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    img_bytes = await image.read()
    base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = image.content_type or "image/jpeg"
    data_url = f"data:{mime_type};base64,{base64_encoded}"
    
    job_id = job_manager.create_job("vlm", {"prompt": prompt, "filename": image.filename}, user_email=user_email)
    background_tasks.add_task(process_vlm_task, job_id, data_url, prompt)
    return {"job_id": job_id}

@app.post("/api/audio/stt")
async def stt_endpoint(background_tasks: BackgroundTasks, audio: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    job_id = job_manager.create_job("stt", {"filename": audio.filename}, user_email=user_email)
    ext = audio.filename.split('.')[-1] if '.' in audio.filename else "mp3"
    filepath = os.path.join(UPLOADS_DIR, f"{job_id}.{ext}")
    with open(filepath, "wb") as f:
        f.write(await audio.read())
        
    background_tasks.add_task(process_stt_task, job_id, filepath)
    return {"job_id": job_id}

@app.post("/api/ppt/generate")
async def generate_ppt_endpoint(background_tasks: BackgroundTasks, topic: str = Form(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    job_id = job_manager.create_job("ppt", {"topic": topic}, user_email=user_email)
    background_tasks.add_task(process_ppt_task, job_id, topic)
    return {"job_id": job_id}

# Job Management Endpoints
@app.get("/api/jobs")
async def get_all_jobs(auth = Depends(flexible_auth)):
    jobs = job_manager.get_jobs()
    user_email = auth.get("email")
    if user_email in ["yeonwoo.kim03@gmail.com", "admin@tor-ai.com"]:
        return jobs
    return {k: v for k, v in jobs.items() if v.get("user_email") == user_email}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, auth = Depends(flexible_auth)):
    jobs = job_manager.get_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    user_email = auth.get("email")
    if user_email not in ["yeonwoo.kim03@gmail.com", "admin@tor-ai.com"] and job.get("user_email") != user_email:
        raise HTTPException(status_code=403, detail="Forbidden")
    return job

@app.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: str, auth = Depends(flexible_auth)):
    jobs = job_manager.get_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    user_email = auth.get("email")
    if user_email not in ["yeonwoo.kim03@gmail.com", "admin@tor-ai.com"] and job.get("user_email") != user_email:
        raise HTTPException(status_code=403, detail="Forbidden")
    job_manager.delete_job(job_id)
    return {"status": "deleted"}

@app.post("/api/gallery/upload")
async def gallery_upload_endpoint(file: UploadFile = File(...), auth = Depends(flexible_auth)):
    user_email = auth.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    ext = file.filename.split('.')[-1] if '.' in file.filename else "dat"
    filename = f"gallery_{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(RESULTS_DIR, filename) # Save to results so it can be served
    with open(filepath, "wb") as f:
        f.write(await file.read())
        
    job_id = job_manager.create_job(
        "gallery_upload", 
        {"instruction": "Direct Upload", "filename": file.filename},
        user_email=user_email
    )
    job_manager.update_job(job_id, "completed", result=f"/api/results/{filename}")
    return {"status": "success", "job_id": job_id, "url": f"/api/results/{filename}"}
@app.get("/api/health/vllm")
async def health_vllm():
    return {"state": gpu_arbiter.state(), "available": gpu_arbiter.vllm_available()}

@app.get("/api/tokens/list")
async def list_tokens():
    return auth_service.get_tokens()

@app.post("/api/tokens/create")
async def create_token(label: str = Form(...)):
    token = auth_service.generate_new_token(label)
    return {"token": token}

@app.get("/api/results/{filename}")
async def get_result_file(filename: str):
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/user/quota")
async def proxy_get_quota(email: str):
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8002/api/user/quota", params={"email": email}) as resp:
            if resp.status != 200:
                return {"credits": {"remaining": 0, "limit": 0}}
            return await resp.json()

# User management endpoints
@app.get("/api/user/info")
async def get_user_info(request: Request):
    # 1. Check headers from Nginx (active when auth_request is used)
    email = request.headers.get("X-Email")
    name = request.headers.get("X-Preferred-Username") or request.headers.get("X-User")
    picture = request.headers.get("X-User-Image")
    
    if email and email not in ["", "None", "Guest"]:
        return {"email": email, "name": name, "picture": picture}
        
    # 2. Try manual verification with cookies (for public landing page check)
    cookie = request.headers.get("Cookie")
    if cookie:
        try:
            async with aiohttp.ClientSession() as session:
                # We pass Host to user_manager so it knows which domain to check
                async with session.get(
                    "http://localhost:8002/auth/verify", 
                    headers={"Cookie": cookie, "Host": request.headers.get("Host", "tor-ai.com")}
                ) as resp:
                    if resp.status == 200:
                        return {
                            "email": resp.headers.get("X-Auth-Request-Email"),
                            "name": resp.headers.get("X-Auth-Request-Preferred-Username") or resp.headers.get("X-Auth-Request-User"),
                            "picture": resp.headers.get("X-Auth-Request-Image")
                        }
        except Exception as e:
            logger.error(f"Manual auth check failed: {e}")

    return {"email": "Guest", "name": "Guest"}

@app.get("/logout")
async def logout(request: Request):
    # Bounce through oauth2-proxy to clear the session cookie, then land on
    # the public /logout_done page (which is exempt from auth_request in
    # nginx) so we don't loop back into the sign-in flow.
    host = request.headers.get("host", "tor-ai.com")
    return RedirectResponse(
        url=f"/oauth2/sign_out?rd=https://{host}/logout_done",
        status_code=303,
    )


@app.get("/logout_done", response_class=HTMLResponse)
async def logout_done(request: Request):
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang=\"ko\"><head>
<meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<title>로그아웃 완료 | TOR-AI</title>
<style>
  body{margin:0;font-family:'Outfit','Pretendard',sans-serif;background:#0f172a;color:#f1f5f9;display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:#1e293b;border:1px solid rgba(255,255,255,.08);border-radius:1.5rem;padding:3rem;max-width:420px;text-align:center;box-shadow:0 25px 50px -12px rgba(0,0,0,.5)}
  h1{margin:0 0 .5rem;font-size:1.5rem}
  p{color:#94a3b8;margin:0 0 2rem}
  .btn{display:inline-block;padding:.875rem 1.5rem;background:#10b981;color:#fff;text-decoration:none;border-radius:.75rem;font-weight:600;transition:background .2s}
  .btn:hover{background:#059669}
  .icon{font-size:3rem;margin-bottom:1rem}
</style></head><body>
<div class=\"card\">
  <div class=\"icon\">👋</div>
  <h1>로그아웃 되었습니다</h1>
  <p>안전하게 세션이 종료되었습니다.</p>
  <a class=\"btn\" href=\"/\">다시 로그인</a>
</div></body></html>""")

@app.post("/api/user/unregister")
async def unregister(request: Request):
    email = request.headers.get("X-Email")
    logger.info(f"User unregister request: {email}")
    return {"status": "success", "redirect": "/oauth2/sign_out"}

@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    user_email = request.headers.get("X-Email", "Guest")
    pref_user = request.headers.get("X-Preferred-Username")
    user_name = pref_user or (user_email.split("@")[0] if "@" in user_email else "User")
    user_image = request.headers.get("X-User-Image") or request.headers.get("X-Auth-Request-Image")
    
    context = {
        "request": request,
        "user_email": user_email,
        "user_name": user_name,
        "user_image": user_image
    }
    return templates.TemplateResponse(request, "landing.html", context)

@app.get("/llm", response_class=HTMLResponse)
async def serve_ui(request: Request):
    try:
        # Detailed Header Logging (sanitized for sensitive info)
        headers = {k: v for k, v in request.headers.items() if "auth" not in k.lower() and "cookie" not in k.lower()}
        logger.info(f"Serve UI - Headers: {headers}")

        user_email = request.headers.get("X-Email")
        if not user_email or user_email in ["", "None", "Guest"]:
            user_email = "Guest"

        # Prefer the human-readable display name from preferred_username
        # (oauth2-proxy + user_manager). X-User is the Google sub (numeric
        # id), so it's only used as a last resort.
        pref_user = request.headers.get("X-Preferred-Username")
        x_user = request.headers.get("X-User")

        def _is_numeric(s: str) -> bool:
            return bool(s) and s.isdigit()

        user_name = pref_user
        if not user_name or _is_numeric(user_name):
            user_name = user_email.split("@")[0] if "@" in user_email and user_email != "Guest" else (x_user or "User")
        if _is_numeric(user_name):
            user_name = "User"

        # Profile picture: user_manager populates this from Google userinfo.
        user_image = request.headers.get("X-User-Image") or request.headers.get("X-Auth-Request-Image")
        if not user_image or user_image in ["", "None", "undefined", "null"]:
            from urllib.parse import quote
            user_image = (
                f"https://ui-avatars.com/api/?name={quote(user_name)}"
                "&background=10b981&color=fff"
            )

        logger.info(f"Resolved User Info - Name: {user_name}, Email: {user_email}, Image: {user_image}")

        context = {
            "request": request,
            "user_email": str(user_email),
            "user_name": str(user_name),
            "user_image": str(user_image)
        }
        return templates.TemplateResponse(request, "index.html", context)
    except Exception as e:
        logger.error(f"Critical Error in serve_ui: {e}", exc_info=True)
        # Fallback to a simple message if template rendering fails
        return HTMLResponse(content=f"<h1>Internal Server Error</h1><p>{str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
