import json
import os
import time

JOB_FILE = "/home/yanus/unified_ai_service/hub_jobs.json"

def get_jobs():
    if not os.path.exists(JOB_FILE):
        return {}
    try:
        with open(JOB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_jobs(jobs):
    with open(JOB_FILE, "w") as f:
        json.dump(jobs, f)

def create_job(job_type: str, input_data: dict) -> str:
    import uuid
    job_id = str(uuid.uuid4())
    jobs = get_jobs()
    jobs[job_id] = {
        "id": job_id,
        "type": job_type,
        "status": "pending",
        "input": input_data,
        "result": None,
        "error": None,
        "created_at": time.time()
    }
    save_jobs(jobs)
    return job_id

def update_job(job_id: str, status: str, result=None, error=None):
    jobs = get_jobs()
    if job_id in jobs:
        jobs[job_id]["status"] = status
        if result is not None:
            jobs[job_id]["result"] = result
        if error is not None:
            jobs[job_id]["error"] = str(error)
        save_jobs(jobs)

def delete_job(job_id: str):
    jobs = get_jobs()
    if job_id in jobs:
        del jobs[job_id]
        save_jobs(jobs)
