import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from document_loader import load_document
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, MAX_DOC_SIZE
from rag import add_document, search, delete_document_chunks, get_stats as rag_stats

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

llm_client: OpenAI | None = None
documents: dict[str, str] = {}  # filename -> text
_doc_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    if DEEPSEEK_API_KEY:
        llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    else:
        logger.warning("未设置 DEEPSEEK_API_KEY")
    # 从磁盘恢复已上传的文档
    upload_dir = "./documents"
    if os.path.exists(upload_dir):
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
                try:
                    documents[fname] = load_document(fpath)
                except Exception as e:
                    logger.warning("恢复文档 %s 失败: %s", fname, e)
    yield


app = FastAPI(title="AI 知识库问答系统", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 不能与 allow_origins=["*"] 同时为 True
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    upload_dir = "./documents"
    os.makedirs(upload_dir, exist_ok=True)
    # 防止路径遍历：只取文件名部分
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="无效的文件名")
    file_path = os.path.join(upload_dir, safe_name)

    content = await file.read()
    if len(content) > MAX_DOC_SIZE:
        raise HTTPException(status_code=400, detail=f"文件过大，最大允许 {MAX_DOC_SIZE // 1024 // 1024}MB")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = load_document(file_path)
        chunk_count = add_document(safe_name, text)  # embedding 成功才算完成
        async with _doc_lock:
            documents[safe_name] = text
        return {"message": "上传成功", "filename": safe_name, "chars": len(text), "chunks": chunk_count}
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {e}")


@app.post("/query", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")
    if not documents:
        return AnswerResponse(answer="还没有上传文档，请先在左侧上传。", sources=[])

    # RAG检索：先找到相关chunk
    hits = await asyncio.to_thread(search, request.question)
    if not hits:
        return AnswerResponse(answer="未找到相关信息，请尝试换个问法。", sources=[])

    # 拼接检索到的内容作为context
    context = "\n\n---\n\n".join(
        f"【文档: {h['filename']}】\n{h['text']}" for h in hits
    )
    # 去重的来源文档名
    source_files = list(dict.fromkeys(h["filename"] for h in hits))

    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个专业的知识库问答助手。请根据以下检索到的文档片段回答用户问题。"
                    "如果文档中没有相关信息，请直接说明。回答时标注信息来源文档。"
                ),
            },
            {"role": "user", "content": f"检索到的文档片段：\n{context}\n\n问题：{request.question}"},
        ],
        max_tokens=2000,
    )

    return AnswerResponse(
        answer=response.choices[0].message.content,
        sources=source_files,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_configured": llm_client is not None,
        "document_count": len(documents),
    }


@app.get("/stats")
async def stats():
    rag_info = rag_stats()
    return {
        "document_count": len(documents),
        "total_chars": sum(len(t) for t in documents.values()),
        "documents": list(documents.keys()),
        "chunk_count": rag_info["chunk_count"],
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    safe_name = os.path.basename(filename)
    async with _doc_lock:
        if safe_name not in documents:
            raise HTTPException(status_code=404, detail="文档不存在")
        del documents[safe_name]
    delete_document_chunks(safe_name)
    file_path = os.path.join("./documents", safe_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": "已删除", "filename": safe_name}


if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0 仅用于开发/演示，生产环境应限制为 127.0.0.1 或由反向代理暴露
    uvicorn.run(app, host="0.0.0.0", port=8000)
