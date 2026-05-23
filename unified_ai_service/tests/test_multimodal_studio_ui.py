from pathlib import Path


TEMPLATE = Path("/home/yanus/unified_ai_service/templates/index.html")


def test_studio_exposes_multimodal_execute_form():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert 'id="multimodalInstruction"' in html
    assert 'id="multimodalFiles"' in html
    assert 'id="multimodalQuality"' in html
    assert 'id="multimodalVoiceProvider"' in html
    assert 'id="multimodalVoice"' in html
    assert "loadElevenLabsVoices()" in html
    assert "runMultimodal()" in html


def test_studio_calls_multimodal_execute_api():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "/api/multimodal/execute" in html
    assert "formData.append('instruction'" in html
    assert "formData.append('quality'" in html
    assert "formData.append('preferred_voice_provider'" in html
    assert "formData.append('preferred_voice'" in html
    assert "formData.append('files'" in html


def test_studio_fetches_elevenlabs_voice_list():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "/api/elevenlabs/voices" in html
    assert "airYK6ydeWdrJg6gyZA3" in html


def test_jobs_render_multimodal_results_as_links_and_plan():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "renderMultimodalResult" in html
    assert "job.type === 'multimodal'" in html
    assert "Multimodal Plan" in html


def test_api_tab_documents_multimodal_usage():
    html = TEMPLATE.read_text(encoding="utf-8")

    assert "Multimodal Execute API" in html
    assert "curl -X POST https://tor-ai.com/api/multimodal/execute" in html
    assert "preferred_voice_provider" in html
