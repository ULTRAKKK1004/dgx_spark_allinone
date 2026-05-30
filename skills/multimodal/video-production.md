# Video Production (`multimodal:video-production`)

## Overview
Guidelines for generating high-quality, professional video content including lecture videos, movie-like scenes, and advertisements.

## Quality Standards
- **Resolution**: Minimum 720p, preferably 1080p for final outputs.
- **Sampling**: Use high-quality samplers (Euler, DPM++ 2M SDE) with sufficient steps (20-30 for base, 4-10 for Wan2.1).
- **Consistency**: Use ControlNet or IP-Adapter to maintain character/style consistency.
- **Noise Reduction**: Ensure appropriate denoise strength (0.6-0.8) during lipsync to avoid "jittery" or "ghosting" effects.

## Workflow: AI Lecture
1. **Base**: High-quality static image (Flux) or subtle idle loop (AnimateDiff).
2. **Audio**: Use ElevenLabs for natural, professional narration.
3. **Sync**: Use LivePortrait for high-res emotional sync, or Wav2Lip-GAN for robust speech alignment.
4. **Post**: Post-process with sharpen filters if needed.

## Mandate
Always verify that the face is clearly visible and centered before starting a lipsync task.
