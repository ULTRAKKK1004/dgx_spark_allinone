import os
import uuid
import base64
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request, BackgroundTasks

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import llm_service
import stt_service
import ppt_service
import auth_service
import job_manager

app = FastAPI(title="DGX Spark AI Hub")

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

# Endpoints
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

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(os.path.join(BASE_DIR, "templates", "index.html"), "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
