# AI Hub Project Instructions

This file contains mandates and guidelines specific to the AI Hub project.

## Skill Integration
This project leverages the **Superpowers** and **Coding** skills installed in `/home/yanus/skills/`.

### Active Skills
- `superpowers:*` (Writing Plans, Executing Plans, Subagent Development)
- `coding:*` (Expert Python, Engineering Standards)
- `multimodal:*` (Video Production, Audio Generation, Image Synthesis)

## Technical Architecture
- **Backend**: FastAPI (Python 3.12) in `unified_ai_service/`.
- **Frontend**: Single Page Application (Tailwind CSS) in `templates/index.html`.
- **Infrastructure**: Docker (ComfyUI, vLLM).
- **Media Engine**: Custom engine in `media_engine/` using `gpu_arbiter`.

## Development Workflow
1. **Plan**: Use `/write-plan` (writing-plans skill).
2. **Execute**: Iterate through tasks, using subagents for parallel work.
3. **Verify**: Always run tests and check mobile responsiveness (100dvh).

## Security
- API tokens must be verified via `auth_service.py`.
- User data must be isolated by email in all queries.
