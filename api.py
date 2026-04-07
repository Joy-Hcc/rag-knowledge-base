# 提供 REST API 接口，供 Streamlit 调用

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from kb.vector_store import VectorStore
from kb.document_loader import load_document
from kb.text_splitter import split_text
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)


# 全局变量
vector_store: VectorStore = None
llm_client: OpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    global vector_store, llm_client

    # 启动时
    vector_store = VectorStore()
    vector_store.create_collection()

    if DEEPSEEK_API_KEY:
        llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
    else:
        print("警告: 未设置 DEEPSEEK_API_KEY，LLM 功能将不可用")

    yield

    # 关闭时（如果需要清理操作可以在这里添加）


app = FastAPI(title="知识库问答 API", lifespan=lifespan)

# CORS 配置，允许 Streamlit 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库"""
    # 保存文件
    upload_dir = "./documents"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 加载并分块
    try:
        text = load_document(file_path)
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档内容为空")

        # 添加到向量库
        ids = [f"doc_{uuid.uuid4().hex[:8]}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.filename, "chunk_id": i} for i in range(len(chunks))]

        vector_store.add_documents(chunks, ids, metadatas)

        return {
            "message": "文档上传成功",
            "filename": file.filename,
            "chunks": len(chunks)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.post("/query", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """基于知识库回答问题"""
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置，请设置 DEEPSEEK_API_KEY")

    # 1. 检索相关文档
    try:
        results = vector_store.search(request.question, TOP_K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

    if not results:
        return AnswerResponse(
            answer="知识库中没有找到相关文档，请先上传文档。",
            sources=[]
        )

    # 2. 构建上下文
    context = "\n\n".join([r["document"] for r in results])

    # 3. 调用大模型生成答案
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的问答助手。请根据提供的上下文内容回答用户的问题。\n\n如果上下文中没有相关信息，请明确告知用户你没有在提供的文档中找到相关内容。"
                },
                {
                    "role": "user",
                    "content": f"上下文内容：\n{context}\n\n用户问题：{request.question}"
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

    return AnswerResponse(
        answer=answer,
        sources=[
            {
                "content": r["document"][:200] + "..." if len(r["document"]) > 200 else r["document"],
                "source": r["metadata"]["source"],
                "chunk_id": r["metadata"].get("chunk_id", 0)
            }
            for r in results
        ]
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    doc_count = vector_store.get_collection_count() if vector_store else 0
    return {
        "status": "ok",
        "llm_configured": llm_client is not None,
        "document_count": doc_count
    }


@app.get("/stats")
async def get_stats():
    """获取知识库统计信息"""
    if vector_store is None:
        return {"count": 0}
    return {"count": vector_store.get_collection_count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
