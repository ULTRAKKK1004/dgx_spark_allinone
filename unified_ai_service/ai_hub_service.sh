#!/bin/bash
cd /home/yanus/unified_ai_service
source venv/bin/activate
exec venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8081
