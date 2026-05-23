import os
import subprocess
import pytest
from pathlib import Path
from media_engine import window


@pytest.fixture
def silent_audio(tmp_path):
    """10초짜리 무음 WAV 생성."""
    from pydub import AudioSegment
    path = tmp_path / "silent.wav"
    AudioSegment.silent(duration=10000).export(path, format="wav")
    return path


@pytest.fixture
def short_video(tmp_path):
    """ffmpeg testsrc로 3초짜리 mp4 생성."""
    path = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=16",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.mark.asyncio
async def test_chunk_audio_fixed_overlap(silent_audio, tmp_path):
    chunks = await window.chunk_audio_fixed(
        str(silent_audio),
        chunk_sec=4,
        overlap_sec=1,
        out_dir=str(tmp_path / "chunks"),
    )
    assert len(chunks) >= 3
    from pydub import AudioSegment
    for c in chunks:
        seg = AudioSegment.from_file(c)
        assert 3500 <= len(seg) <= 4500


@pytest.mark.asyncio
async def test_chunk_audio_smart_silence_split(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine
    seg = (Sine(440).to_audio_segment(duration=2000) +
           AudioSegment.silent(duration=1000) +
           Sine(440).to_audio_segment(duration=2000) +
           AudioSegment.silent(duration=1000) +
           Sine(440).to_audio_segment(duration=2000))
    src = tmp_path / "speech.wav"
    seg.export(src, format="wav")

    chunks = await window.chunk_audio_smart(
        str(src), target_range=(2, 6), out_dir=str(tmp_path / "out")
    )
    assert 2 <= len(chunks) <= 5


@pytest.mark.asyncio
async def test_extract_last_frame(short_video, tmp_path):
    out = await window.extract_last_frame(str(short_video), out_dir=str(tmp_path))
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


@pytest.mark.asyncio
async def test_concat_videos(short_video, tmp_path):
    out = tmp_path / "joined.mp4"
    await window.concat_videos([str(short_video), str(short_video)], str(out))
    assert out.exists()
    dur = await window.get_media_duration(str(out))
    assert 5.5 < dur < 6.5


@pytest.mark.asyncio
async def test_crossfade_audio(silent_audio, tmp_path):
    out = tmp_path / "xf.wav"
    await window.crossfade_audio_segments(
        [str(silent_audio), str(silent_audio)],
        overlap_ms=500,
        output_path=str(out),
    )
    assert out.exists()
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(out)
    assert 19000 < len(seg) < 20000
