from media_capabilities import SUPPORTED_ACTIONS, get_action, planner_prompt


def test_supported_actions_include_initial_multimodal_surface():
    assert "text.extract" in SUPPORTED_ACTIONS
    assert "image.generate" in SUPPORTED_ACTIONS
    assert "image.analyze" in SUPPORTED_ACTIONS
    assert "voice.tts" in SUPPORTED_ACTIONS
    assert "ppt.generate" in SUPPORTED_ACTIONS
    assert "video.analyze" in SUPPORTED_ACTIONS
    assert "package.bundle" in SUPPORTED_ACTIONS


def test_get_action_returns_io_contract():
    action = get_action("voice.tts")

    assert action["kind"] == "voice"
    assert "text" in action["inputs"]
    assert action["outputs"] == {"audio": "audio file url"}


def test_planner_prompt_is_compact_and_mentions_schema_rules():
    prompt = planner_prompt()

    assert "Return JSON only" in prompt
    assert "voice.tts" in prompt
    assert "video.lipsync" in prompt
    assert "Unsupported actions are invalid" in prompt
