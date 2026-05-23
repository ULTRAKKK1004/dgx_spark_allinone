from pathlib import Path

import pytest

import voice_providers


def test_choose_provider_auto_without_key_uses_local(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    assert voice_providers.choose_provider("auto", quality="high") == "local_f5"


def test_choose_provider_auto_with_key_and_high_quality_uses_elevenlabs(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice")

    assert voice_providers.choose_provider("auto", quality="high") == "elevenlabs"


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
