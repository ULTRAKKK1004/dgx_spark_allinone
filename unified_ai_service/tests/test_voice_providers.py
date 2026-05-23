from pathlib import Path

import pytest

import voice_providers


def test_default_elevenlabs_voice_id_is_user_requested_value(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)

    assert voice_providers.get_elevenlabs_voice_id() == "airYK6ydeWdrJg6gyZA3"


def test_choose_provider_auto_without_key_uses_local(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    assert voice_providers.choose_provider("auto", quality="high") == "local_f5"


def test_choose_provider_auto_with_key_and_high_quality_uses_elevenlabs(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")

    assert voice_providers.choose_provider("auto", quality="high") == "elevenlabs"


def test_choose_provider_auto_with_key_and_missing_local_uses_elevenlabs(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setattr(voice_providers, "local_f5_available", lambda: False)

    assert voice_providers.choose_provider("auto", quality="draft") == "elevenlabs"


def test_choose_provider_auto_without_key_and_missing_local_keeps_local_for_clear_error(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(voice_providers, "local_f5_available", lambda: False)

    assert voice_providers.choose_provider("auto", quality="draft") == "local_f5"


def test_split_text_preserves_all_content():
    text = "첫 문장입니다. 둘째 문장입니다.\n\n새 문단입니다."

    chunks = voice_providers.split_text_for_tts(text, limit=12)

    assert "".join(chunk.text for chunk in chunks).replace("\n\n", "") == text.replace("\n\n", "")
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_synthesize_auto_falls_back_to_local(monkeypatch, tmp_path):
    async def fake_local(text, ref_audio, ref_text, output_path):
        Path(output_path).write_bytes(b"local wav")
        return output_path

    async def fake_eleven(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setattr(voice_providers, "_synthesize_local_f5", fake_local)
    monkeypatch.setattr(voice_providers, "_synthesize_elevenlabs", fake_eleven)

    result = await voice_providers.synthesize_speech(
        "긴 강의 대본",
        provider="auto",
        quality="high",
        output_path=str(tmp_path / "out.wav"),
    )

    assert result["provider"] == "local_f5"
    assert result["url"].endswith("out.wav") or result["path"].endswith("out.wav")


@pytest.mark.asyncio
async def test_synthesize_forced_elevenlabs_failure_raises(monkeypatch, tmp_path):
    async def fake_eleven(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")
    monkeypatch.setattr(voice_providers, "_synthesize_elevenlabs", fake_eleven)

    with pytest.raises(RuntimeError, match="network down"):
        await voice_providers.synthesize_speech(
            "text",
            provider="elevenlabs",
            output_path=str(tmp_path / "out.mp3"),
        )


@pytest.mark.asyncio
async def test_list_elevenlabs_voices_returns_compact_voice_list(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "voices": [
                    {
                        "voice_id": "v1",
                        "name": "Voice One",
                        "category": "cloned",
                        "labels": {"language": "ko"},
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            assert url == "https://api.elevenlabs.io/v1/voices"
            assert headers["xi-api-key"] == "key"
            return FakeResponse()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setattr(voice_providers.httpx, "AsyncClient", FakeClient)

    voices = await voice_providers.list_elevenlabs_voices()

    assert voices == [{"voice_id": "v1", "name": "Voice One", "category": "cloned", "labels": {"language": "ko"}}]
