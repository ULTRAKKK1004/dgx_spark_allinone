# Audio Generation (`multimodal:audio-generation`)

## Overview
Workflow for creating music, background tracks, and professional voiceovers.

## Music Generation
- **Engine**: Use Audiocraft (MusicGen) for local generation.
- **Parameters**: 
  - `duration`: default 10-30s. 
  - `top_k`: 250, `top_p`: 0, `temperature`: 1.0.
- **Quality**: Ensure the output is saved in high-bitrate WAV or MP3.

## Voice Generation (TTS)
- **Primary**: ElevenLabs (Paid). Use for high-stakes narration and lectures.
- **Secondary**: Local F5-TTS/Bark. Use for draft or cost-sensitive tasks.
- **Styles**: Always match the voice style (calm, energetic, professional) to the content prompt.

## Mandate
For AI Hub, **only ElevenLabs** is approved as a paid external service. All other generation should be local.
