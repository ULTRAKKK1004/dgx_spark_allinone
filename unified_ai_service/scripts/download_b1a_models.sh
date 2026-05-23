#!/usr/bin/env bash
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/home/yanus/Docker/models}"
NODES_ROOT="${NODES_ROOT:-/home/yanus/Docker/custom_nodes_persistent}"

mkdir -p "$MODELS_ROOT/diffusion_models/FLUX1"
mkdir -p "$MODELS_ROOT/controlnet/FLUX"
mkdir -p "$NODES_ROOT"

echo "=== [1/3] FLUX dev fp8 (~17GB) ==="
FLUX_PATH="$MODELS_ROOT/diffusion_models/FLUX1/flux1-dev-fp8.safetensors"
if [ -f "$FLUX_PATH" ] && [ "$(stat -c%s "$FLUX_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$FLUX_PATH" | cut -f1))"
else
  wget --continue -O "$FLUX_PATH.partial" \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
  mv "$FLUX_PATH.partial" "$FLUX_PATH"
  echo "  downloaded: $(du -h "$FLUX_PATH" | cut -f1)"
fi

echo "=== [2/3] FLUX-ControlNet-Union-Pro (~3GB) ==="
CN_PATH="$MODELS_ROOT/controlnet/FLUX/FLUX.1-dev-ControlNet-Union-Pro.safetensors"
if [ -f "$CN_PATH" ] && [ "$(stat -c%s "$CN_PATH")" -gt 1000000000 ]; then
  echo "  already present ($(du -h "$CN_PATH" | cut -f1))"
else
  wget --continue -O "$CN_PATH.partial" \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors"
  mv "$CN_PATH.partial" "$CN_PATH"
  echo "  downloaded: $(du -h "$CN_PATH" | cut -f1)"
fi

echo "=== [3/3] ComfyUI-controlnet-aux custom node ==="
AUX_DIR="$NODES_ROOT/ComfyUI-controlnet-aux"
if [ -d "$AUX_DIR/.git" ]; then
  echo "  already cloned"
else
  git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux "$AUX_DIR"
  echo "  -> ComfyUI 컨테이너 재시작 필요: docker restart comfyui"
fi

echo ""
echo "=== Done ==="
