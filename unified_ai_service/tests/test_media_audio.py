from pathlib import Path

from pydub import AudioSegment

import media_audio


def test_write_silent_sfx_creates_valid_wav(tmp_path):
    path = tmp_path / "fallback.wav"

    media_audio._write_silent_sfx(str(path), duration_ms=250)

    assert path.exists()
    assert len(AudioSegment.from_wav(path)) == 250
