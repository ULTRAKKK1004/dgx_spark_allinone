"""워크플로우 Jinja 템플릿이 ComfyUI API JSON으로 정상 렌더되는지 확인."""
import json
import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from media_engine import catalog


WF_DIR = Path(__file__).parent.parent / "workflows"


def _render(template: str, params: dict) -> dict:
    env = Environment(
        loader=FileSystemLoader(str(WF_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",)),
    )
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    rendered = env.get_template(template).render(**params)
    return json.loads(rendered)


def test_zimage_turbo_renders_valid_json():
    meta = catalog.get("image.gen.zimage_turbo")
    params = catalog.validate(meta, {"prompt": "a quiet harbor at dusk"})
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "a quiet harbor at dusk" in flat


def test_zimage_turbo_special_chars_escape():
    """프롬프트에 따옴표/개행이 있어도 JSON이 깨지지 않는다."""
    meta = catalog.get("image.gen.zimage_turbo")
    params = catalog.validate(meta, {"prompt": 'a "tall" cat\nwith \\backslash'})
    wf = _render(meta["template"], params)
    flat = json.dumps(wf)
    assert "tall" in flat


def test_qwen_edit_renders_valid_json():
    meta = catalog.get("image.edit.qwen")
    params = catalog.validate(meta, {
        "prompt": "make it night time",
        "image_name": "uploaded_xyz.png",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "make it night time" in flat
    assert "uploaded_xyz.png" in flat


def test_qwen_edit_denoise_applied():
    meta = catalog.get("image.edit.qwen")
    params = catalog.validate(meta, {
        "prompt": "p",
        "image_name": "x.png",
        "denoise": 0.5,
    })
    wf = _render(meta["template"], params)
    sampler_node = next(
        v for v in wf.values()
        if v.get("class_type") == "KSampler"
    )
    assert abs(sampler_node["inputs"]["denoise"] - 0.5) < 1e-6


def test_wan22_i2v_renders_valid_json():
    meta = catalog.get("video.i2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "camera slowly pans right",
        "image_name": "start_frame.png",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "camera slowly pans right" in flat
    assert "start_frame.png" in flat


def test_wan22_i2v_frames_match():
    meta = catalog.get("video.i2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "p",
        "image_name": "x.png",
        "frames": 65,
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf)
    assert '"length": 65' in flat or '"length":65' in flat


def test_wan22_s2v_renders_valid_json():
    meta = catalog.get("video.s2v.wan22")
    params = catalog.validate(meta, {
        "prompt": "speaker in front of bookshelf",
        "image_name": "ref.png",
        "audio_name": "speech.wav",
    })
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "speech.wav" in flat
    assert "ref.png" in flat


def test_flux_renders_valid_json():
    meta = catalog.get("image.gen.flux")
    params = catalog.validate(meta, {"prompt": "a futuristic city skyline"})
    wf = _render(meta["template"], params)
    assert isinstance(wf, dict)
    assert meta["output_node"] in wf
    flat = json.dumps(wf, ensure_ascii=False)
    assert "a futuristic city skyline" in flat
    assert "flux1-dev-fp8.safetensors" in flat


def test_flux_ctrl_union_canny():
    meta = catalog.get("image.ctrl.flux_union")
    params = catalog.validate(meta, {
        "prompt": "pose-controlled portrait",
        "control_image": "ctrl_in.png",
        "control_type": "canny",
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf, ensure_ascii=False)
    assert "ctrl_in.png" in flat
    assert "FLUX.1-dev-ControlNet-Union-Pro.safetensors" in flat


def test_flux_ctrl_union_maps_api_type_to_comfyui_enum():
    meta = catalog.get("image.ctrl.flux_union")
    params = catalog.validate(meta, {
        "prompt": "edge-controlled portrait",
        "control_image": "ctrl_in.png",
        "control_type": "canny",
    })
    wf = _render(meta["template"], params)
    assert wf["6"]["inputs"]["type"] == "canny/lineart/anime_lineart/mlsd"


def test_flux_ctrl_union_openpose_changes_mode():
    meta = catalog.get("image.ctrl.flux_union")
    p_canny = catalog.validate(meta, {"prompt": "p", "control_image": "x.png", "control_type": "canny"})
    p_pose = catalog.validate(meta, {"prompt": "p", "control_image": "x.png", "control_type": "openpose"})
    wf_canny = _render(meta["template"], p_canny)
    wf_pose = _render(meta["template"], p_pose)
    flat_canny = json.dumps(wf_canny)
    flat_pose = json.dumps(wf_pose)
    assert flat_canny != flat_pose, "control_type 변경이 워크플로우에 반영되어야 함"


def test_qwen_inpaint_renders_valid_json():
    meta = catalog.get("image.inpaint.qwen")
    params = catalog.validate(meta, {
        "prompt": "replace background with beach",
        "image_name": "img.png",
        "mask_name": "msk.png",
    })
    wf = _render(meta["template"], params)
    flat = json.dumps(wf, ensure_ascii=False)
    assert "img.png" in flat
    assert "msk.png" in flat
    assert "LoadImageMask" in flat
