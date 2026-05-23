import pytest

from multimodal_models import MediaAsset, MediaPlan, PlanValidationError


SUPPORTED = {"voice.tts", "image.generate", "package.bundle"}


def test_media_plan_validates_minimal_plan():
    plan = MediaPlan.from_dict(
        {
            "version": "1",
            "goal": "강의 음성 생성",
            "quality": "high",
            "steps": [
                {
                    "id": "step_1",
                    "action": "voice.tts",
                    "inputs": {"text": "안녕하세요"},
                    "outputs": {"audio": "lecture_voice"},
                }
            ],
            "final": {"primary": "lecture_voice", "format": "audio"},
        },
        supported_actions=SUPPORTED,
        asset_aliases=set(),
    )

    assert plan.goal == "강의 음성 생성"
    assert plan.steps[0].action == "voice.tts"
    assert plan.final["primary"] == "lecture_voice"


def test_media_plan_rejects_unknown_action():
    with pytest.raises(PlanValidationError, match="unknown action"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {
                        "id": "step_1",
                        "action": "video.teleport",
                        "inputs": {},
                        "outputs": {"video": "out"},
                    }
                ],
                "final": {"primary": "out", "format": "video"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_rejects_duplicate_step_ids():
    with pytest.raises(PlanValidationError, match="duplicate step id"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {"id": "same", "action": "voice.tts", "inputs": {"text": "a"}, "outputs": {"audio": "a"}},
                    {"id": "same", "action": "voice.tts", "inputs": {"text": "b"}, "outputs": {"audio": "b"}},
                ],
                "final": {"primary": "b", "format": "audio"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_rejects_duplicate_output_aliases():
    with pytest.raises(PlanValidationError, match="duplicate output alias"):
        MediaPlan.from_dict(
            {
                "version": "1",
                "goal": "bad",
                "steps": [
                    {"id": "step_1", "action": "voice.tts", "inputs": {"text": "a"}, "outputs": {"audio": "same"}},
                    {"id": "step_2", "action": "image.generate", "inputs": {"prompt": "cat"}, "outputs": {"image": "same"}},
                ],
                "final": {"primary": "same", "format": "image"},
            },
            supported_actions=SUPPORTED,
            asset_aliases=set(),
        )


def test_media_plan_accepts_uploaded_asset_alias_reference():
    plan = MediaPlan.from_dict(
        {
            "version": "1",
            "goal": "image",
            "steps": [
                {
                    "id": "step_1",
                    "action": "image.generate",
                    "inputs": {"prompt": "based on asset:image_1"},
                    "outputs": {"image": "out_image"},
                }
            ],
            "final": {"primary": "out_image", "format": "image"},
        },
        supported_actions=SUPPORTED,
        asset_aliases={"image_1"},
    )

    assert plan.steps[0].inputs["prompt"] == "based on asset:image_1"


def test_media_asset_public_url_for_results_file():
    asset = MediaAsset(alias="voice", path="/home/yanus/unified_ai_service/results/a.wav", mime_type="audio/wav")

    assert asset.public_url() == "/api/results/a.wav"
