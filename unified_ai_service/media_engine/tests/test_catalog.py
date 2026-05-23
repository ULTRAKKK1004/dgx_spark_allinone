import pytest
from media_engine import catalog


def test_get_returns_workflow_metadata():
    """등록된 워크플로우는 메타데이터를 반환한다."""
    meta = catalog.get("image.gen.zimage_turbo")
    assert meta["template"] == "image_gen_zimage_turbo.json.j2"
    assert meta["vram_class"] == "light"
    assert meta["output_node"]  # 비어있지 않음
    assert meta["timeout_sec"] >= 30


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown.workflow"):
        catalog.get("unknown.workflow")


def test_validate_fills_defaults():
    meta = catalog.get("image.gen.zimage_turbo")
    out = catalog.validate(meta, {"prompt": "a cat"})
    assert out["prompt"] == "a cat"
    assert out["width"] == 1024  # default
    assert out["steps"] == 8     # turbo default


def test_validate_rejects_missing_required():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="prompt"):
        catalog.validate(meta, {})


def test_validate_coerces_types():
    meta = catalog.get("image.gen.zimage_turbo")
    out = catalog.validate(meta, {"prompt": "x", "width": "768"})
    assert out["width"] == 768  # str → int


def test_validate_rejects_invalid_type():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="width"):
        catalog.validate(meta, {"prompt": "x", "width": "not-a-number"})


def test_list_workflows_returns_all_ids():
    ids = catalog.list_workflows()
    assert "image.gen.zimage_turbo" in ids
    assert "image.edit.qwen" in ids
    assert "video.i2v.wan22" in ids
    assert "video.s2v.wan22" in ids


def test_check_models_present_returns_missing_list(tmp_path, monkeypatch):
    """모델 디렉토리가 비면 모든 필수 모델이 누락으로 보고된다."""
    monkeypatch.setattr(catalog, "MODELS_ROOT", str(tmp_path))
    meta = catalog.get("image.gen.zimage_turbo")
    missing = catalog.check_models_present(meta)
    assert len(missing) > 0
    assert all(isinstance(m, str) for m in missing)


def test_validate_rejects_unknown_params():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="unknown params"):
        catalog.validate(meta, {"prompt": "x", "wdith": 768})


def test_validate_rejects_none_value():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="cannot be None"):
        catalog.validate(meta, {"prompt": None})


def test_validate_rejects_bool_for_int():
    meta = catalog.get("image.gen.zimage_turbo")
    with pytest.raises(ValueError, match="bool not accepted"):
        catalog.validate(meta, {"prompt": "x", "width": True})


def test_flux_workflow_metadata():
    meta = catalog.get("image.gen.flux")
    assert meta["template"] == "image_gen_flux.json.j2"
    assert meta["vram_class"] == "heavy"
    assert meta["output_node"] == "9"
    assert meta["timeout_sec"] >= 300


def test_flux_ctrl_union_metadata():
    meta = catalog.get("image.ctrl.flux_union")
    assert meta["template"] == "image_ctrl_flux_union.json.j2"
    assert meta["vram_class"] == "heavy"
    params = catalog.validate(meta, {
        "prompt": "p",
        "control_image": "ctrl.png",
    })
    assert params["control_type"] == "canny"
    assert params["strength"] == 0.7


def test_qwen_inpaint_metadata():
    meta = catalog.get("image.inpaint.qwen")
    params = catalog.validate(meta, {
        "prompt": "remove the cat",
        "image_name": "img.png",
        "mask_name": "msk.png",
    })
    assert params["denoise"] == 0.9


def test_list_workflows_includes_b1a_additions():
    ids = set(catalog.list_workflows())
    assert {"image.gen.flux", "image.ctrl.flux_union", "image.inpaint.qwen"}.issubset(ids)


def test_flux_ctrl_union_rejects_unknown_control_type():
    meta = catalog.get("image.ctrl.flux_union")
    with pytest.raises(ValueError, match="control_type"):
        catalog.validate(meta, {
            "prompt": "p",
            "control_image": "ctrl.png",
            "control_type": "not-real",
        })
