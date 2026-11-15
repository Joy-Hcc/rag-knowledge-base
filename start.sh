#!/usr/bin/env bash
# 启动后端 + 前端。后端只监听 127.0.0.1:8000，浏览器访问 3000，由 Next.js 的 /api 代理转发。
set -e
cd "$(dirname "$0")"
mkdir -p logs

pkill -f "uvicorn api:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
sleep 1

setsid nohup .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 > logs/backend.log 2>&1 &
(cd frontend && setsid nohup env BACKEND_URL=http://127.0.0.1:8000 npm start > ../logs/frontend.log 2>&1 &)
sleep 5

curl -sf localhost:8000/health && echo
curl -sf -o /dev/null localhost:3000/ && echo "前端已就绪：http://localhost:3000"
