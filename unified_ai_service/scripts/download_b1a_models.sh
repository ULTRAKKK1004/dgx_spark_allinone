#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/home/yanus/Docker/models}"
NODES_ROOT="${NODES_ROOT:-/home/yanus/Docker/custom_nodes_persistent}"

mkdir -p "$MODELS_ROOT/diffusion_models/FLUX1"
mkdir -p "$MODELS_ROOT/controlnet/FLUX"
mkdir -p "$MODELS_ROOT/text_encoders"
mkdir -p "$NODES_ROOT"

echo "=== [1/5] FLUX dev fp8 (~17GB) ==="
FLUX_PATH="$MODELS_ROOT/diffusion_models/FLUX1/flux1-dev-fp8.safetensors"
if [ -f "$FLUX_PATH" ] && [ "$(stat -c%s "$FLUX_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$FLUX_PATH" | cut -f1))"
else
  wget --continue -O "$FLUX_PATH.partial" \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
  mv "$FLUX_PATH.partial" "$FLUX_PATH"
  echo "  downloaded: $(du -h "$FLUX_PATH" | cut -f1)"
fi

echo "=== [2/5] FLUX CLIP-L text encoder (~234MB) ==="
CLIP_PATH="$MODELS_ROOT/text_encoders/clip_l.safetensors"
if [ -f "$CLIP_PATH" ] && [ "$(stat -c%s "$CLIP_PATH")" -gt 100000000 ]; then
  echo "  already present ($(du -h "$CLIP_PATH" | cut -f1))"
else
  wget --continue -O "$CLIP_PATH.partial" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
  mv "$CLIP_PATH.partial" "$CLIP_PATH"
  echo "  downloaded: $(du -h "$CLIP_PATH" | cut -f1)"
fi

echo "=== [3/5] FLUX T5XXL fp8 text encoder (~4.6GB) ==="
T5_PATH="$MODELS_ROOT/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
if [ -f "$T5_PATH" ] && [ "$(stat -c%s "$T5_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$T5_PATH" | cut -f1))"
else
  wget --continue -O "$T5_PATH.partial" \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
  mv "$T5_PATH.partial" "$T5_PATH"
  echo "  downloaded: $(du -h "$T5_PATH" | cut -f1)"
fi

echo "=== [4/5] FLUX-ControlNet-Union-Pro (~6GB) ==="
CN_PATH="$MODELS_ROOT/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors"
if [ -f "$CN_PATH" ] && [ "$(stat -c%s "$CN_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$CN_PATH" | cut -f1))"
else
  wget --continue -O "$CN_PATH.partial" \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"
  mv "$CN_PATH.partial" "$CN_PATH"
  echo "  downloaded: $(du -h "$CN_PATH" | cut -f1)"
fi

echo "=== [5/5] ComfyUI-controlnet-aux custom node ==="
AUX_DIR="$NODES_ROOT/ComfyUI-controlnet-aux"
if [ -d "$AUX_DIR/.git" ]; then
  echo "  already cloned"
else
  git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux "$AUX_DIR"
  echo "  -> ComfyUI 컨테이너 재시작 필요: docker restart comfyui"
fi

echo ""
echo "=== Done ==="
