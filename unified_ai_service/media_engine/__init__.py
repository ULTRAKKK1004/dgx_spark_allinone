"""media_engine — Phase A 공용 미디어 엔진.

Submodules:
- catalog: 워크플로우 메타데이터 단일 출처
- comfyui_client: ComfyUI HTTP I/O
- gpu_arbiter: vLLM ↔ ComfyUI heavy 모델 직렬화
- window: 무빙윈도우 청크/병합 유틸
- runner: 통합 실행 진입점 (render → submit → poll → fetch)
- job_queue: 잡 관리자 위의 직렬화 래퍼
"""

__version__ = "0.1.0"
