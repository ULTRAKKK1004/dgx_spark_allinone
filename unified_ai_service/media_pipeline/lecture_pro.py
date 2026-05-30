import asyncio
import os
import uuid
import logging
from typing import List, Dict, Any
from pathlib import Path
from moviepy import ImageClip, VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
import json

import llm_service
from media_engine import gpu_arbiter, window
import media_audio
from render_service import SlideRenderer

logger = logging.getLogger(__name__)

BASE_DIR = "/home/yanus/unified_ai_service"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TMP_DIR = os.path.join(BASE_DIR, "tmp_video")

async def _split_script_per_slide(script: str, num_slides: int) -> List[str]:
    """Uses LLM to optimally split the script into parts matching the number of slides."""
    if num_slides == 1:
        return [script]
        
    prompt = f"""
    Split the following script into EXACTLY {num_slides} logical parts to match a presentation with {num_slides} slides.
    Return ONLY a JSON array of strings, where each string is the script for one slide.
    Ensure the array length is exactly {num_slides}.
    Script:
    {script}
    """
    try:
        response = await llm_service.generate_text(prompt, "You are a helpful JSON parser.")
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        parts = json.loads(response.strip())
        if isinstance(parts, list) and len(parts) > 0:
            # Pad or trim if LLM failed to match exactly
            while len(parts) < num_slides:
                parts.append("Continuing to the next slide.")
            return parts[:num_slides]
    except Exception as e:
        logger.error(f"Script split failed: {e}")
    
    # Naive split fallback
    words = script.split()
    chunk_size = max(1, len(words) // num_slides)
    return [" ".join(words[i*chunk_size : (i+1)*chunk_size]) for i in range(num_slides)]


async def generate_pro_lecture(topic: str, script: str, face_image: str) -> str:
    """Generates a long-form PPT-synced lecture video with PiP talking head."""
    logger.info("Starting Pro Lecture Pipeline for topic: %s", topic)
    
    # 1. Generate PPT Slides via LLM + HTML
    sys_prompt = """You are a master presentation designer. 
    Generate a high-fidelity 16:9 HTML presentation about the topic. 
    Make exactly 5 slides for a comprehensive lecture.
    Each slide MUST be a <div class="slide"> with fixed dimensions (1920x1080px).
    Use Tailwind CSS (via CDN) for styling. 
    Output ONLY the complete HTML content."""
    
    html_content = await llm_service.generate_text(f"Create a 5-slide presentation on: {topic}", sys_prompt)
    if html_content.startswith("```html"): html_content = html_content[7:-3]
    
    renderer = SlideRenderer(TMP_DIR)
    slide_images = await renderer.render_html_to_images(html_content.strip())
    
    if not slide_images:
        raise ValueError("Failed to generate slide images.")
        
    num_slides = len(slide_images)
    logger.info("Generated %d slide images.", num_slides)
    
    # 2. Split script
    script_parts = await _split_script_per_slide(script, num_slides)
    
    # 3. Generate Audio and Video per slide
    final_clips = []
    
    # Create the idle base once (to save time, we reuse the same idle base for all lipsyncing)
    # To keep it memory efficient and fast, we use the original face_image for lipsync 
    # instead of a heavy idle loop, because LivePortrait animates still images perfectly well!
    
    for i in range(num_slides):
        slide_img = slide_images[i]
        slide_text = script_parts[i]
        
        logger.info("Slide %d: TTS generation...", i+1)
        audio_path = await media_audio.generate_tts_with_effects(slide_text, None, "")
        
        duration = await window.get_media_duration(audio_path)
        logger.info("Slide %d: Lip-syncing %f seconds...", i+1, duration)
        
        # We process lipsync in smart chunks inside `lipsync_video` or just pass the full audio 
        # (LivePortrait handles small videos well, but since we are doing PiP, we can resize the input face first to save VRAM).
        from media_video import lipsync_video
        
        # Lipsync the talking head
        head_video = await lipsync_video(face_image, audio_path, workflow="video.lipsync.liveportrait")
        
        # Composite using MoviePy
        slide_clip = ImageClip(slide_img).with_duration(duration)
        head_clip = VideoFileClip(head_video).resize(height=360).margin(10, color=(255,255,255)).with_position(("right", "bottom"))
        audio_clip = AudioFileClip(audio_path)
        
        composite = CompositeVideoClip([slide_clip, head_clip]).with_audio(audio_clip)
        final_clips.append(composite)
        
    # 4. Concatenate
    logger.info("Concatenating %d clips...", len(final_clips))
    final_video = concatenate_videoclips(final_clips)
    
    output_filename = f"lecture_pro_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    final_video.write_videofile(
        output_path, 
        fps=24, 
        codec="libx264", 
        audio_codec="aac",
        logger=None
    )
    
    # Cleanup
    for c in final_clips: c.close()
    final_video.close()
    
    logger.info("Pro Lecture completed: %s", output_path)
    return output_path
