import os
import uuid
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import llm_service
import stt_service
import ppt_service
import auth_service

app = FastAPI(title="DGX Spark AI Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    """
    Allows access if:
    1. The request comes from the Web UI (no Authorization header, but referred by local domain)
    2. A valid Bearer token is provided in the Authorization header
    """
    # If Authorization header is present, it MUST be valid
    if authorization:
        token = authorization.replace("Bearer ", "")
        if auth_service.verify_token(token):
            return token
        raise HTTPException(status_code=401, detail="Invalid API Token")
    
    # If no Authorization header, check if it's likely a browser request from the same origin
    referer = request.headers.get("referer", "")
    host = request.headers.get("host", "")
    
    if host in referer or not referer:
        # We allow web UI requests without tokens
        return "web_ui_session"
        
    raise HTTPException(status_code=401, detail="Missing Authorization Header for API call")

class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful AI assistant running on DGX Spark."

@app.post("/api/llm/chat")
async def chat_endpoint(req: ChatRequest, auth = Depends(flexible_auth)):
    try:
        response_text = await llm_service.generate_text(req.prompt, req.system_prompt)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vlm/analyze")
async def analyze_image_endpoint(prompt: str = Form(...), image: UploadFile = File(...), auth = Depends(flexible_auth)):
    try:
        img_bytes = await image.read()
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = image.content_type or "image/jpeg"
        data_url = f"data:{mime_type};base64,{base64_encoded}"
        response_text = await llm_service.analyze_image(data_url, prompt)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/stt")
async def stt_endpoint(audio: UploadFile = File(...), auth = Depends(flexible_auth)):
    file_id = str(uuid.uuid4())
    ext = audio.filename.split('.')[-1] if '.' in audio.filename else "mp3"
    filepath = os.path.join(UPLOADS_DIR, f"{file_id}.{ext}")
    with open(filepath, "wb") as f:
        f.write(await audio.read())
    try:
        text = stt_service.transcribe_audio(filepath)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ppt/generate")
async def generate_ppt_endpoint(topic: str = Form(...), auth = Depends(flexible_auth)):
    try:
        slides_data = await llm_service.generate_ppt_structure(topic)
        file_id = str(uuid.uuid4())
        output_filename = f"presentation_{file_id}.pptx"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        ppt_service.generate_ppt_file(title=topic, slides_data=slides_data, output_path=output_path)
        return {"download_url": f"api/results/{output_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
