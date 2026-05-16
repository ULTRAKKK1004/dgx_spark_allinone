import os
import json
import uuid
import asyncio
import traceback
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Header, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .orchestrator import LectureOrchestrator
from dotenv import load_dotenv

BASE_DIR = "/home/yanus/lecture_service"
load_dotenv(os.path.join(BASE_DIR, ".env"))
EXPECTED_TOKEN = os.getenv("ACCESS_TOKEN")

app = FastAPI()

async def verify_token(authorization: str = Header(None)):
    if EXPECTED_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
             raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ")[1]
        if token != EXPECTED_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid access token")

JOBS_FILE = os.path.join(BASE_DIR, "jobs.json")
UPLOADS_DIR = "/home/yanus/uploads"
RESULTS_DIR = "/home/yanus/results"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

jobs = {}

def load_jobs():
    global jobs
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                jobs = json.load(f)
        except Exception as e:
            print(f"DEBUG: Failed to load jobs: {e}")
            jobs = {}

def save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f)
    except Exception as e:
        print(f"DEBUG: Failed to save jobs: {e}")

load_jobs()

# UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Lecture Pro - Wan2.2 Edition</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; background-color: #0e1117; color: #e6edf3; }
        .container { background: #161b22; padding: 40px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        h1 { color: #58a6ff; text-align: center; margin-bottom: 30px; font-weight: 300; }
        .form-group { margin-bottom: 25px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #8b949e; }
        input[type="file"], input[type="text"], textarea { width: 100%; padding: 12px; border: 1px solid #30363d; border-radius: 6px; background: #0d1117; color: #c9d1d9; box-sizing: border-box; }
        textarea { height: 100px; resize: vertical; }
        .btn-group { display: flex; gap: 10px; margin-top: 20px; }
        button { background: #238636; color: white; border: none; padding: 15px 30px; border-radius: 6px; cursor: pointer; font-size: 18px; font-weight: 600; width: 100%; transition: background 0.2s; }
        button:hover { background: #2ea043; }
        button:disabled { background: #444; cursor: not-allowed; }
        button.reset { background: #da3633; width: auto; font-size: 14px; padding: 10px 20px; border:none; border-radius:6px; color:white; cursor:pointer;}
        button.reset:hover { background: #f85149; }
        #activeJobs { margin-top: 40px; }
        .job-card { border: 1px solid #30363d; padding: 20px; border-radius: 8px; margin-bottom: 20px; background: #0d1117; }
        .job-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 15px; }
        video { margin-top: 10px; width: 100%; border-radius: 8px; border: 1px solid #30363d; }
        .previews { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; margin-top: 15px; }
        .status-badge { padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; background: #112747; color: #58a6ff; transition: all 0.3s; }
        .status-completed { background: #11341a; color: #3fb950; }
        .status-failed { background: #341111; color: #f85149; }
        .status-starting { background: #333; color: #aaa; }
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h1>🎓 AI Lecture Pro (Wan2.2)</h1>
            <button class="reset" onclick="resetTasks()">Reset All History</button>
        </div>
        
        <div id="uploadForm">
            <div class="form-group">
                <label>1. Lecturer Photo (Portrait)</label>
                <input type="file" id="imageInput" accept="image/*">
            </div>
            <div class="form-group">
                <label>2. Lecture Audio (Long speech)</label>
                <input type="file" id="audioInput" accept="audio/*">
            </div>
            <div class="form-group">
                <label>3. Lecture Topic/Prompt (Optional)</label>
                <input type="text" id="promptInput" placeholder="e.g. Physics lecture about black holes">
            </div>
            <div class="form-group" style="display:flex; align-items:center; gap:10px;">
                <input type="checkbox" id="directImage" style="width:auto;">
                <label for="directImage" style="margin:0;">Use my photo directly (Skip AI enhancement)</label>
            </div>
            <p style="color: #8b949e; font-size: 14px;">* The system will automatically generate a professional lecture hall setting.</p>
            <button id="submitBtn" onclick="generateVideo()">Create Lecture Video</button>
        </div>

        <div id="activeJobs">
            <h2 style="color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 10px;">Tasks</h2>
            <div id="jobsList"></div>
        </div>
    </div>

    <script>
        let pollingJobs = new Set();
        let seenChunks = {};

        async function loadJobs() {
            try {
                const resp = await fetch('/jobs');
                const jobs = await resp.json();
                const list = document.getElementById('jobsList');
                
                const jobIds = Object.keys(jobs).sort((a, b) => b.localeCompare(a));
                
                if (jobIds.length === 0) {
                    list.innerHTML = '<p id="no-task-msg" style="text-align:center; color:#8b949e;">No tasks yet.</p>';
                    return;
                }

                const noTaskMsg = document.getElementById('no-task-msg');
                if (noTaskMsg) noTaskMsg.remove();

                jobIds.forEach(jobId => {
                    updateOrCreateJobCard(jobId, jobs[jobId]);
                    if (jobs[jobId].status !== 'completed' && jobs[jobId].status !== 'failed') {
                        startPolling(jobId);
                    }
                });
            } catch (e) { console.error("Failed to load jobs", e); }
        }

        function updateOrCreateJobCard(jobId, job) {
            const list = document.getElementById('jobsList');
            let card = document.getElementById(`card-${jobId}`);
            if (!card) {
                card = document.createElement('div');
                card.id = `card-${jobId}`;
                card.className = 'job-card';
                list.prepend(card);
            }
            
            let statusClass = '';
            if (job.status === 'completed') statusClass = 'status-completed';
            else if (job.status === 'failed') statusClass = 'status-failed';
            else if (job.status === 'starting') statusClass = 'status-starting';

            const contentHtml = job.status === 'completed' ? `<video controls src="${job.result}"></video>` : 
                                job.status === 'failed' ? `<p style="color:#f85149">Error: ${job.error}</p>` : 
                                `<div id="previews-${jobId}" class="previews"></div>`;

            card.innerHTML = `
                <div class="job-header">
                    <span><strong>ID:</strong> ${jobId.substring(0,8)}</span>
                    <span id="badge-${jobId}" class="status-badge ${statusClass}">${job.status}</span>
                </div>
                <div id="content-${jobId}">
                    ${contentHtml}
                </div>
            `;
            
            if (job.chunks) {
                job.chunks.forEach((url, i) => addChunkToCard(jobId, url, i));
            }
        }

        function addChunkToCard(jobId, url, index) {
            if (!seenChunks[jobId]) seenChunks[jobId] = new Set();
            if (seenChunks[jobId].has(url)) return;
            seenChunks[jobId].add(url);
            
            const pArea = document.getElementById(`previews-${jobId}`);
            if (pArea) {
                const div = document.createElement('div');
                div.innerHTML = `<p style="font-size:10px;margin:0;color:#8b949e">Chunk ${index+1}</p><video autoplay loop muted playsinline style="width:100%; border-radius:4px; border:1px solid #30363d;" src="${url}"></video>`;
                pArea.appendChild(div);
            }
        }

        function startPolling(jobId) {
            if (pollingJobs.has(jobId)) return;
            pollingJobs.add(jobId);
            
            const interval = setInterval(async () => {
                try {
                    const resp = await fetch(`status/${jobId}`);
                    if (resp.status === 404) { clearInterval(interval); pollingJobs.delete(jobId); return; }
                    const data = await resp.json();
                    
                    const badge = document.getElementById(`badge-${jobId}`);
                    if (badge) {
                        badge.innerText = data.status;
                        badge.className = 'status-badge';
                        if (data.status === 'completed') badge.classList.add('status-completed');
                        else if (data.status === 'failed') badge.classList.add('status-failed');
                        else if (data.status === 'starting') badge.classList.add('status-starting');
                    }

                    if (data.chunks) {
                        data.chunks.forEach((url, i) => addChunkToCard(jobId, url, i));
                    }

                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(interval);
                        pollingJobs.delete(jobId);
                        updateOrCreateJobCard(jobId, data);
                    }
                } catch (e) { console.error("Polling error", e); }
            }, 3000);
        }

        async function generateVideo() {
            const imageFile = document.getElementById('imageInput').files[0];
            const audioFile = document.getElementById('audioInput').files[0];
            const prompt = document.getElementById('promptInput').value;
            const useDirectImage = document.getElementById('directImage').checked;
            const submitBtn = document.getElementById('submitBtn');

            if (!imageFile || !audioFile) { alert("Select files first."); return; }

            submitBtn.disabled = true;
            submitBtn.innerText = "Uploading...";
            
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('audio', audioFile);
            formData.append('prompt', prompt || "");
            formData.append('use_direct_image', useDirectImage);

            try {
                const response = await fetch('/generate', { method: 'POST', body: formData });
                const data = await response.json();
                
                if (data.job_id) {
                    updateOrCreateJobCard(data.job_id, {status: 'starting', chunks: []});
                    startPolling(data.job_id);
                } else if (data.detail) {
                    alert("Error: " + data.detail);
                }
            } catch (e) { 
                alert("Server request failed."); 
            }
            
            submitBtn.disabled = false;
            submitBtn.innerText = "Start Generation";
        }

        async function resetTasks() {
            if (confirm("Reset all tasks?")) {
                await fetch('/reset', { method: 'POST' });
                location.reload();
            }
        }

        loadJobs();
    </script>
</body>
</html>
"""

@app.get("/jobs")
async def list_jobs():
    return jobs

@app.post("/reset")
async def reset():
    global jobs
    jobs = {}
    if os.path.exists(JOBS_FILE): os.remove(JOBS_FILE)
    return {"status": "reset"}

@app.post("/generate")
async def generate(background_tasks: BackgroundTasks, 
                   image: UploadFile = File(...), 
                   audio: UploadFile = File(...), 
                   prompt: str = Form(""), 
                   use_direct_image: bool = Form(False)):
    job_id = str(uuid.uuid4())
    print(f"DEBUG: Starting job {job_id}")
    
    # Secure filename and absolute paths
    img_filename = f"{job_id}_{image.filename.replace(' ', '_')}"
    aud_filename = f"{job_id}_{audio.filename.replace(' ', '_')}"
    img_path = os.path.join(UPLOADS_DIR, img_filename)
    aud_path = os.path.join(UPLOADS_DIR, aud_filename)
    
    try:
        with open(img_path, "wb") as f: f.write(await image.read())
        with open(aud_path, "wb") as f: f.write(await audio.read())
        print(f"DEBUG: Files saved for {job_id}")
    except Exception as e:
        err = f"File save failed: {traceback.format_exc()}"
        print(f"DEBUG: {err}")
        return JSONResponse(status_code=500, content={"detail": err})
    
    jobs[job_id] = {"status": "starting", "result": None, "error": None, "chunks": []}
    save_jobs()
    
    def status_cb(msg, chunk_url=None):
        try:
            if job_id in jobs and jobs[job_id]["status"] not in ["completed", "failed"]:
                jobs[job_id]["status"] = msg
                if chunk_url:
                    if chunk_url not in jobs[job_id]["chunks"]:
                        jobs[job_id]["chunks"].append(chunk_url)
                save_jobs()
        except: pass

    print(f"DEBUG: Dispatching background task for {job_id}")
    background_tasks.add_task(run_orchestrator, job_id, img_path, aud_path, prompt, use_direct_image, status_cb)
    
    return {"job_id": job_id}

async def run_orchestrator(job_id, img_path, aud_path, prompt, use_direct_image, status_cb):
    try:
        orchestrator = LectureOrchestrator(job_id, img_path, aud_path, prompt, use_direct_image)
        result_path = await orchestrator.process(status_callback=status_cb)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = f"results/{job_id}_final.mp4"
        save_jobs()
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"ERROR in job {job_id}:\n{error_msg}")
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            save_jobs()

@app.get("/", response_class=HTMLResponse)
async def index(): return HTML_TEMPLATE

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job: return JSONResponse(status_code=404, content={"detail": "Job not found"})
    return job

@app.get("/results/{filename:path}")
async def get_result(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path): return FileResponse(path)
    return JSONResponse(status_code=404, content={"detail": "File not found"})

@app.get("/chunk/{job_id}/{filename:path}")
async def get_chunk(job_id: str, filename: str):
    path = os.path.join(RESULTS_DIR, job_id, filename)
    if os.path.exists(path): return FileResponse(path)
    return JSONResponse(status_code=404, content={"detail": "Chunk not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
