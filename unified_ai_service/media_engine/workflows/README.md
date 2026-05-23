# media_engine 워크플로우 카탈로그

각 `.json.j2` 파일은 ComfyUI API 형식 JSON에 Jinja2 변수만 노출한 템플릿이다.
파라미터·메타데이터는 `../catalog.py` 의 `WORKFLOWS` 딕셔너리에서 관리한다.

## 신규 워크플로우 추가 절차

1. ComfyUI 웹 UI에서 워크플로우를 만들고 **"Save (API Format)"** 으로 export.
2. `<id>.json.j2` 로 이 디렉토리에 저장.
3. 가변 부분(prompt, image_name, seed, steps 등)을 `{{ name | tojson }}` (문자열) 또는 `{{ name }}` (숫자)으로 치환.
4. `catalog.py` 의 `WORKFLOWS` 에 항목 추가:
   - `template`, `params` (타입+기본값), `models_required`, `output_node`, `vram_class`, `timeout_sec`
5. `test_workflow_render.py` 에 렌더 테스트 추가.

## 현재 등록 워크플로우

- `image_gen_zimage_turbo.json.j2` — z_image_turbo (light, ~5초)
- `image_edit_qwen.json.j2` — Qwen Image Edit 2509 img2img/edit (heavy, ~60초)
- `video_i2v_wan22.json.j2` — Wan2.2 i2v 14B, 5초 청크 (heavy, ~5분)
- `video_s2v_wan22.json.j2` — Wan2.2 s2v 14B 립싱크 (heavy, ~7분)
