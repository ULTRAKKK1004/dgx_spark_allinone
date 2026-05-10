#!/bin/bash

# Script to switch between latest models
MODEL=$1

if [ "$MODEL" == "qwen" ]; then
    echo "Switching to Qwen 3.6 (27B Dense)..."
    sed -i 's|google/gemma-4-26B-A4B-it|Qwen/Qwen3.6-27B-Instruct|g' /home/yanus/Docker/docker-compose.vllm.yml
    sed -i 's|google/gemma-4-26B-A4B-it|Qwen/Qwen3.6-27B-Instruct|g' /home/yanus/unified_ai_service/llm_service.py
    sed -i 's|Gemma-4|Qwen-3.6|g' /home/yanus/unified_ai_service/templates/index.html
elif [ "$MODEL" == "gemma" ]; then
    echo "Switching to Gemma 4 (26B)..."
    sed -i 's|Qwen/Qwen3.6-27B-Instruct|google/gemma-4-26B-A4B-it|g' /home/yanus/Docker/docker-compose.vllm.yml
    sed -i 's|Qwen/Qwen3.6-27B-Instruct|google/gemma-4-26B-A4B-it|g' /home/yanus/unified_ai_service/llm_service.py
    sed -i 's|Qwen-3.6|Gemma-4|g' /home/yanus/unified_ai_service/templates/index.html
else
    echo "Usage: ./switch_model.sh [gemma|qwen]"
    exit 1
fi

cd /home/yanus/Docker
docker compose -f docker-compose.vllm.yml down
docker compose -f docker-compose.vllm.yml up -d
pkill -f "python3 -m uvicorn main:app"
cd /home/yanus/unified_ai_service
nohup venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8081 > uvicorn.log 2>&1 &

echo "Model switched and services restarted. Initial download may take time."
