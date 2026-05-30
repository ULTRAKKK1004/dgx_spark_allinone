# Image Synthesis (`multimodal:image-synthesis`)

## Overview
Optimizing text-to-image and image-to-image workflows for maximum prompt adherence and visual quality.

## Model Selection
- **Flux (Dev/Schnell)**: Best for complex prompts, text rendering, and photorealism.
- **ZImage-Turbo**: Optimized for speed (8 steps). Use for instant drafts or simple concepts.
- **Janus-Pro**: Use for vision-language tasks (analysis) or interleaved generation.

## Prompt Engineering
- **Detail**: Explicitly mention lighting (cinematic, soft), style (photorealistic, 8k), and composition (wide shot, close-up).
- **Negative Prompts**: Use to eliminate common artifacts (deformed hands, blurry faces) in SDXL-based workflows.

## Mandate
If the user's prompt is complex or contains specific text, default to **Flux** regardless of speed settings.
