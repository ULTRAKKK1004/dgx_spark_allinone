import os
import json
import uuid
import asyncio
import aiohttp
import subprocess
import time
import shutil
from .workflows import get_stage1_image_gen_workflow, get_stage2_animatediff_workflow, get_wan_s2v_workflow

COMFYUI_URL = "http://localhost:8188"
# These should match the host paths mapped in docker-compose
HOST_INPUT_DIR = "/home/yanus/Docker/input"
HOST_OUTPUT_DIR = "/home/yanus/Docker/output"

class LectureOrchestrator:
    def __init__(self, job_id, lecturer_img, audio_path, prompt, use_direct_image=False):
        self.job_id = job_id
        self.lecturer_img = lecturer_img
        self.audio_path = audio_path
        self.prompt = prompt
        self.use_direct_image = use_direct_image
        self.chunks = []
        self.results_dir = f"results/{job_id}"
        os.makedirs(self.results_dir, exist_ok=True)

    def log(self, msg):
        print(f"[{self.job_id}] {msg}")
        # Always append to a global log for debugging
        with open("/home/yanus/lecture_service/service.log", "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{self.job_id}] {msg}\n")

    async def upload_file(self, file_path, filename):
        self.log(f"Uploading {filename} to ComfyUI (from {file_path})...")
        if not os.path.exists(file_path):
            self.log(f"ERROR: File not found for upload: {file_path}")
            raise Exception(f"File not found: {file_path}")
        
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field('image', f, filename=filename)
                async with session.post(f"{COMFYUI_URL}/upload/image", data=data) as resp:
                    res_json = await resp.json()
                    if resp.status != 200:
                        self.log(f"Upload FAILED: {res_json}")
                        raise Exception(f"Upload failed: {res_json}")
                    self.log(f"Upload SUCCESS: {res_json}")
                    return res_json

    async def run_workflow(self, prompt_json):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_json}) as resp:
                data = await resp.json()
                if "prompt_id" not in data:
                    self.log(f"ComfyUI API Rejected Prompt: {data}")
                    raise Exception(f"ComfyUI API Rejected Prompt: {data}")
                prompt_id = data["prompt_id"]
            
            self.log(f"Workflow started. Prompt ID: {prompt_id}")
            while True:
                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as h_resp:
                    history = await h_resp.json()
                    if prompt_id in history:
                        node_history = history[prompt_id]
                        if "status" in node_history and node_history["status"].get("status_str") == "error":
                            self.log(f"Workflow execution FAILED: {node_history['status']}")
                            raise Exception(f"Workflow execution failed: {node_history['status']}")
                        
                        outputs = node_history.get("outputs", {})
                        for node_id in outputs:
                            if 'gifs' in outputs[node_id]: return outputs[node_id]['gifs'][0]['filename']
                            if 'images' in outputs[node_id]: return outputs[node_id]['images'][0]['filename']
                            if 'videos' in outputs[node_id]: return outputs[node_id]['videos'][0]['filename']
                await asyncio.sleep(3)

    def slice_audio(self):
        self.log("Smart slicing audio based on silence detection...")
        from pydub import AudioSegment, silence
        
        audio = AudioSegment.from_file(self.audio_path)
        # Find silences (min_silence_len in ms, silence_thresh in dBFS)
        dbfs = audio.dBFS
        silent_ranges = silence.detect_silence(audio, min_silence_len=300, silence_thresh=dbfs-16)
        
        split_points = [0]
        for start, end in silent_ranges:
            split_points.append((start + end) / 2)
        split_points.append(len(audio))
        
        # Target 5-7s chunks for Wan2.2 stability
        final_splits = [0]
        last_split = 0
        for p in split_points:
            duration = p - last_split
            if duration >= 4000: # at least 4s
                if duration <= 8000:
                    final_splits.append(p)
                    last_split = p
                else:
                    while (p - last_split) > 8000:
                        force_split = last_split + 6000
                        final_splits.append(force_split)
                        last_split = force_split
                    final_splits.append(p)
                    last_split = p
        
        if final_splits[-1] < len(audio):
            final_splits.append(len(audio))
            
        final_splits = sorted(list(set(final_splits)))
        
        self.chunks = []
        for i in range(len(final_splits) - 1):
            start_t = final_splits[i]
            end_t = final_splits[i+1]
            chunk = audio[start_t:end_t]
            chunk_name = f"chunk_{i:03d}.mp3"
            chunk_path = os.path.join(self.results_dir, chunk_name)
            chunk.export(chunk_path, format="mp3")
            self.chunks.append(chunk_name)
            
        self.log(f"Smart sliced into {len(self.chunks)} chunks.")

    def get_audio_duration(self, path):
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    async def process(self, status_callback=None):
        def update_status(msg):
            self.log(msg)
            if status_callback:
                status_callback(msg)

        # Stage 1: Image Gen + Face Swap
        update_status("Stage 1: Generating lecturer image and swapping face...")
        img_name = f"lecturer_input_{self.job_id}.jpg"
        await self.upload_file(self.lecturer_img, img_name)
        
        wf_s1 = get_stage1_image_gen_workflow(img_name, self.prompt, self.job_id)
        s1_img_filename = await self.run_workflow(wf_s1)
        s1_img_path = os.path.join(HOST_OUTPUT_DIR, s1_img_filename)
        
        # Copy S1 image to results for reference
        shutil.copy(s1_img_path, os.path.join(self.results_dir, "stage1_output.jpg"))
        
        # Stage 2: AnimateDiff Idle Loop
        update_status("Stage 2: Creating idle loop with AnimateDiff...")
        await self.upload_file(os.path.join(self.results_dir, "stage1_output.jpg"), f"s1_output_{self.job_id}.jpg")
        
        wf_s2 = get_stage2_animatediff_workflow(f"s1_output_{self.job_id}.jpg", self.job_id)
        s2_vid_filename = await self.run_workflow(wf_s2)
        s2_vid_path = os.path.join(HOST_OUTPUT_DIR, s2_vid_filename)
        
        # Copy S2 video to results
        shutil.copy(s2_vid_path, os.path.join(self.results_dir, "stage2_output.mp4"))
        
        # Extract first frame of S2 as starting point for S3
        current_ref_path = os.path.join(self.results_dir, "ref_0.jpg")
        cmd = ["ffmpeg", "-y", "-i", s2_vid_path, "-vframes", "1", "-q:v", "1", current_ref_path]
        subprocess.run(cmd, check=True)
        
        current_ref_name = f"ref_start_{self.job_id}.jpg"
        await self.upload_file(current_ref_path, current_ref_name)
        current_ref_image = current_ref_name

        # Stage 3: Lip-Sync (Wan2.2)
        update_status("Stage 3: Slicing audio and performing lip-sync...")
        self.slice_audio()
        
        chunk_videos = []
        for i, chunk in enumerate(self.chunks):
            update_status(f"Generating Video Chunk {i+1}/{len(self.chunks)}...")
            chunk_path = os.path.join(self.results_dir, chunk)
            chunk_duration = self.get_audio_duration(chunk_path)
            await self.upload_file(chunk_path, chunk)
            
            wf_s3 = get_wan_s2v_workflow(current_ref_image, chunk, self.prompt, self.job_id, i, chunk_duration)
            video_filename = await self.run_workflow(wf_s3)
            video_path = os.path.join(HOST_OUTPUT_DIR, video_filename)
            chunk_videos.append(video_path)
            
            # Extract last frame for next chunk
            if i < len(self.chunks) - 1:
                next_ref_path = f"{self.results_dir}/ref_{i+1}.jpg"
                cmd = ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path, "-update", "1", "-q:v", "1", next_ref_path]
                subprocess.run(cmd, capture_output=True)
                
                next_ref_name = f"ref_{self.job_id}_{i+1}.jpg"
                await self.upload_file(next_ref_path, next_ref_name)
                current_ref_image = next_ref_name

        # Final Concatenation
        update_status("Concatenating final video...")
        concat_file = f"{self.results_dir}/concat.txt"
        with open(concat_file, "w") as f:
            for v in chunk_videos:
                if os.path.exists(v):
                    f.write(f"file '{v}'\n")
        
        final_video_path = f"results/{self.job_id}_final.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final_video_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        update_status("Job complete!")
        return final_video_path
