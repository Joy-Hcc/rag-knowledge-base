# 后端：FastAPI + RAG
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py rag.py config.py document_loader.py ./

# 数据目录（chromadb 向量库 + 上传的文档），建议用 volume 持久化
VOLUME ["/app/chroma_db", "/app/documents"]

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
