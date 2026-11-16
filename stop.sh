#!/usr/bin/env bash
# 停止后端与前端
pkill -f "uvicorn api:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
pkill -f "next-server" 2>/dev/null || true
echo "已停止（日志保留在 logs/）"
