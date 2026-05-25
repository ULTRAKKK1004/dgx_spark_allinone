import pytest
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from media_video import generate_lecture_video, lipsync_video

@pytest.mark.asyncio
async def test_generate_lecture_video_flow():
    """lecture_video 생성이 올바른 순서로 서브 함수들을 호출하는지 테스트."""
    with patch("media_image.generate_image", new_callable=AsyncMock) as mock_gen, \
         patch("media_video._face_swap", new_callable=AsyncMock) as mock_swap, \
         patch("media_video._generate_idle_loop", new_callable=AsyncMock) as mock_idle, \
         patch("media_video.lipsync_video", new_callable=AsyncMock) as mock_lip, \
         patch("media_engine.window.chunk_audio_smart", new_callable=AsyncMock) as mock_chunk, \
         patch("media_engine.window.get_media_duration", new_callable=AsyncMock) as mock_dur, \
         patch("media_engine.window.concat_videos", new_callable=AsyncMock) as mock_concat, \
         patch("media_video._ensure_audio_duration_matches", new_callable=AsyncMock) as mock_sync, \
         patch("shutil.copy") as mock_copy, \
         patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        
        mock_gen.return_value = Path("/tmp/base.png")
        mock_swap.return_value = Path("/tmp/swapped.png")
        mock_idle.return_value = Path("/tmp/idle.mp4")
        mock_lip.return_value = Path("/tmp/final.mp4")
        mock_chunk.return_value = [Path("/tmp/chunk1.wav")]
        mock_dur.return_value = 5.0
        mock_sync.side_effect = lambda v, a: v
        
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_exec.return_value = mock_proc
        
        result = await generate_lecture_video(
            face_image_path="/api/results/face.png",
            audio_path="/api/results/audio.wav",
            prompt="physics"
        )
        
        assert result.name.startswith("lecture_final_")
        assert str(result.parent).endswith("results")
        mock_gen.assert_called_once()
        mock_swap.assert_called_once()
        mock_idle.assert_called_once()
        mock_lip.assert_called_once()
        # LivePortrait가 기본으로 설정되었는지 확인
        assert mock_lip.call_args[1]["workflow"] == "video.lipsync.liveportrait"

@pytest.mark.asyncio
async def test_lipsync_video_upload_logic():
    """lipsync_video가 파일을 업로드하고 runner를 호출하는지 테스트."""
    with patch("media_engine.comfyui_client.upload_image", new_callable=AsyncMock) as mock_up_img, \
         patch("media_engine.comfyui_client.upload_audio", new_callable=AsyncMock) as mock_up_aud, \
         patch("media_engine.runner.run", new_callable=AsyncMock) as mock_run, \
         patch("os.path.exists", return_value=True):
        
        mock_run.return_value = Path("/tmp/out.mp4")
        
        await lipsync_video("/tmp/in.mp4", "/tmp/aud.wav")
        
        mock_up_img.assert_called_once()
        mock_up_aud.assert_called_once()
        mock_run.assert_called_once()
