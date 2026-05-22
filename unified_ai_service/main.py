import os
import uuid
import base64
import asyncio
import logging
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
            return token
        logger.warning(f"Invalid API Token: {token[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API Token")
    
    referer = request.headers.get("referer", "")
    host = request.headers.get("host", "")
    logger.info(f"Auth Check - Host: {host}, Referer: {referer}")
    
    if (host and host in referer) or not referer:
        return "web_ui_session"
    
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

def process_stt_task(job_id: str, filepath: str):
    try:
        job_manager.update_job(job_id, "processing")
        text = stt_service.transcribe_audio(filepath)
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


async def process_media_image_task(job_id: str, prompt: str):
    try:
        job_manager.update_job(job_id, "processing")
        result_path = await media_image.generate_image(prompt)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_media_music_task(job_id: str, prompt: str, duration: int):
    try:
        job_manager.update_job(job_id, "processing")
        result_path = await media_audio.generate_music(prompt, duration)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_media_tts_task(job_id: str, text: str, ref_audio: str, ref_text: str):
    try:
        job_manager.update_job(job_id, "processing")
        result_path = await media_audio.generate_tts_with_effects(text, ref_audio, ref_text)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_media_video_gen_task(job_id: str, prompt: str, base_image: str, target_dur: int):
    try:
        job_manager.update_job(job_id, "processing")
        result_path = await media_video.generate_long_video(prompt, base_image, target_dur)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_media_video_edit_task(job_id: str, video_path: str, audio_path: str, prompt: str):
    try:
        job_manager.update_job(job_id, "processing")
        result_path = await media_video.edit_video(video_path, audio_path, prompt=prompt)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))


async def process_media_video_shorts_task(job_id: str, video_path: str, prompt: str):
    try:
        job_manager.update_job(job_id, "processing")
        import media_video
        result_path = await media_video.shorten_video(video_path, prompt)
        job_manager.update_job(job_id, "completed", result=f"/api/results/{os.path.basename(result_path)}")
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

async def process_media_video_analyze_task(job_id: str, video_path: str, prompt: str):
    try:
        job_manager.update_job(job_id, "processing")
        import media_video
        result_text = await media_video.analyze_video(video_path, prompt)
        job_manager.update_job(job_id, "completed", result=result_text)
    except Exception as e:
        job_manager.update_job(job_id, "failed", error=str(e))

# Endpoints

@app.post("/api/media/image")
async def generate_image_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(...), auth = Depends(flexible_auth)):
    job_id = job_manager.create_job("media_image", {"prompt": prompt})
    background_tasks.add_task(process_media_image_task, job_id, prompt)
    return {"job_id": job_id}

@app.post("/api/media/music")
async def generate_music_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(...), duration: int = Form(10), auth = Depends(flexible_auth)):
    job_id = job_manager.create_job("media_music", {"prompt": prompt, "duration": duration})
    background_tasks.add_task(process_media_music_task, job_id, prompt, duration)
    return {"job_id": job_id}

@app.post("/api/media/tts")
async def generate_tts_endpoint(background_tasks: BackgroundTasks, text: str = Form(...), ref_audio: UploadFile = File(None), ref_text: str = Form(""), auth = Depends(flexible_auth)):
    ref_path = ""
    if ref_audio:
        ref_path = os.path.join(UPLOADS_DIR, f"ref_{uuid.uuid4().hex}_{ref_audio.filename}")
        with open(ref_path, "wb") as f:
            f.write(await ref_audio.read())
            
    job_id = job_manager.create_job("media_tts", {"text": text})
    background_tasks.add_task(process_media_tts_task, job_id, text, ref_path, ref_text)
    return {"job_id": job_id}

@app.post("/api/media/video/gen")
async def generate_video_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(...), duration: int = Form(30), base_image: UploadFile = File(...), auth = Depends(flexible_auth)):
    img_path = os.path.join(UPLOADS_DIR, f"base_img_{uuid.uuid4().hex}_{base_image.filename}")
    with open(img_path, "wb") as f:
        f.write(await base_image.read())
        
    job_id = job_manager.create_job("media_video_gen", {"prompt": prompt, "duration": duration})
    background_tasks.add_task(process_media_video_gen_task, job_id, prompt, img_path, duration)
    return {"job_id": job_id}

@app.post("/api/media/video/edit")
async def edit_video_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(""), video: UploadFile = File(...), audio: UploadFile = File(None), auth = Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"edit_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
        
    aud_path = ""
    if audio:
        aud_path = os.path.join(UPLOADS_DIR, f"edit_aud_{uuid.uuid4().hex}_{audio.filename}")
        with open(aud_path, "wb") as f:
            f.write(await audio.read())
            
    job_id = job_manager.create_job("media_video_edit", {"prompt": prompt})
    background_tasks.add_task(process_media_video_edit_task, job_id, vid_path, aud_path, prompt)
    return {"job_id": job_id}


@app.post("/api/media/video/shorts")
async def video_shorts_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(""), video: UploadFile = File(...), auth = Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"shorts_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
        
    job_id = job_manager.create_job("media_video_shorts", {"prompt": prompt})
    background_tasks.add_task(process_media_video_shorts_task, job_id, vid_path, prompt)
    return {"job_id": job_id}

@app.post("/api/media/video/analyze")
async def video_analyze_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(""), video: UploadFile = File(...), auth = Depends(flexible_auth)):
    vid_path = os.path.join(UPLOADS_DIR, f"analyze_vid_{uuid.uuid4().hex}_{video.filename}")
    with open(vid_path, "wb") as f:
        f.write(await video.read())
        
    job_id = job_manager.create_job("media_video_analyze", {"prompt": prompt})
    background_tasks.add_task(process_media_video_analyze_task, job_id, vid_path, prompt)
    return {"job_id": job_id}

@app.post("/api/llm/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks, auth = Depends(flexible_auth)):
    job_id = job_manager.create_job("llm", {"prompt": req.prompt})
    background_tasks.add_task(process_llm_task, job_id, req)
    return {"job_id": job_id}

@app.post("/api/vlm/analyze")
async def analyze_image_endpoint(background_tasks: BackgroundTasks, prompt: str = Form(...), image: UploadFile = File(...), auth = Depends(flexible_auth)):
    img_bytes = await image.read()
    base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = image.content_type or "image/jpeg"
    data_url = f"data:{mime_type};base64,{base64_encoded}"
    
    job_id = job_manager.create_job("vlm", {"prompt": prompt, "filename": image.filename})
    background_tasks.add_task(process_vlm_task, job_id, data_url, prompt)
    return {"job_id": job_id}

@app.post("/api/audio/stt")
async def stt_endpoint(background_tasks: BackgroundTasks, audio: UploadFile = File(...), auth = Depends(flexible_auth)):
    job_id = job_manager.create_job("stt", {"filename": audio.filename})
    ext = audio.filename.split('.')[-1] if '.' in audio.filename else "mp3"
    filepath = os.path.join(UPLOADS_DIR, f"{job_id}.{ext}")
    with open(filepath, "wb") as f:
        f.write(await audio.read())
        
    background_tasks.add_task(process_stt_task, job_id, filepath)
    return {"job_id": job_id}

@app.post("/api/ppt/generate")
async def generate_ppt_endpoint(background_tasks: BackgroundTasks, topic: str = Form(...), auth = Depends(flexible_auth)):
    job_id = job_manager.create_job("ppt", {"topic": topic})
    background_tasks.add_task(process_ppt_task, job_id, topic)
    return {"job_id": job_id}

# Job Management Endpoints
@app.get("/api/jobs")
async def get_all_jobs(auth = Depends(flexible_auth)):
    return job_manager.get_jobs()

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, auth = Depends(flexible_auth)):
    jobs = job_manager.get_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: str, auth = Depends(flexible_auth)):
    job_manager.delete_job(job_id)
    return {"status": "deleted"}

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

# User management endpoints
@app.get("/api/user/info")
async def get_user_info(request: Request):
    email = request.headers.get("X-Email")
    user = request.headers.get("X-User")
    return {"email": email, "name": user}

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
