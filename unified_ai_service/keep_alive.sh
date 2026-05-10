#!/bin/bash
while true; do
  curl -s http://localhost:8081/ > /dev/null
  if [ $? -ne 0 ]; then
    pkill -f "uvicorn main:app"
    nohup /home/yanus/unified_ai_service/ai_hub_service.sh > /home/yanus/unified_ai_service/uvicorn.log 2>&1 &
  fi
  
  docker inspect -f '{{.State.Running}}' vllm-server 2>/dev/null | grep -q "true"
  if [ $? -ne 0 ]; then
    cd /home/yanus/Docker
    docker compose -f docker-compose.vllm.yml up -d
  fi
  
  sleep 600
done
