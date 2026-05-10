#!/bin/bash

# DGX Spark - vLLM Startup Script (Docker Edition)
# Starts an OpenAI-compatible vLLM server using the local vllm-node image.

DOCKER_COMPOSE_FILE="/home/yanus/Docker/docker-compose.vllm.yml"

echo "🚀 Starting vLLM server via Docker..."
echo "Model: google/gemma-2-9b-it"

# Note: We are using Gemma-2-9b as it is fully compatible and fast.
# Ensure your HF_TOKEN is set in /home/yanus/Docker/.env if you want to swap models.

cd /home/yanus/Docker
docker compose -f "$DOCKER_COMPOSE_FILE" up -d

echo "-------------------------------------------------------"
echo "Server is starting in the background."
echo "You can check logs with: docker logs -f vllm-server"
echo "Note: Downloading Gemma-4 (26B) will take significant time on first run."
echo "-------------------------------------------------------"
