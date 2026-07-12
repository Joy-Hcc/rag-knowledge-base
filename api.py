import os
import json
import asyncio
import logging
import uuid
import queue
from contextlib import asynccontextmanager
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI

from document_loader import load_document
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, MAX_DOC_SIZE,
    MAX_HISTORY_TURNS, CORS_ORIGINS,
)
from rag import add_document, search, delete_document_chunks, get_stats as rag_stats, _build_bm25_index

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

llm_client: OpenAI | None = None
documents: dict[str, str] = {}  # filename -> text
_doc_lock = asyncio.Lock()

# ── 对话历史管理 ──────────────────────────────────────
# conversation_id -> deque of {role, content}，最多保留 MAX_HISTORY_TURNS * 2 条消息
_conversations: OrderedDict[str, list[dict]] = OrderedDict()
MAX_CONVERSATIONS = 100  # 内存中最多保留的会话数（LRU 淘汰）


def _get_history(conversation_id: str) -> list[dict]:
    """获取对话历史，并标记为最近使用"""
    if conversation_id in _conversations:
        # 移到末尾（最近使用）
        _conversations.move_to_end(conversation_id)
        return _conversations[conversation_id]
    return []


def _append_history(conversation_id: str, role: str, content: str):
    """追加一条对话历史"""
    if conversation_id not in _conversations:
        # LRU 淘汰：超过上限时删除最早的会话
        if len(_conversations) >= MAX_CONVERSATIONS:
            _conversations.popitem(last=False)
        _conversations[conversation_id] = []
    else:
        # 标记为最近使用
        _conversations.move_to_end(conversation_id)
    _conversations[conversation_id].append({"role": role, "content": content})
    # 截断：保留最近 MAX_HISTORY_TURNS * 2 条（每轮 = user + assistant）
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(_conversations[conversation_id]) > max_msgs:
        _conversations[conversation_id] = _conversations[conversation_id][-max_msgs:]


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
    # 启动时重建 BM25 索引
    _build_bm25_index()
    yield


app = FastAPI(title="AI 知识库问答系统", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str | None = None  # 可选，用于多轮对话


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    conversation_id: str


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
        text = await asyncio.to_thread(load_document, file_path)
        chunk_count = await asyncio.to_thread(add_document, safe_name, text)
        async with _doc_lock:
            documents[safe_name] = text
        return {"message": "上传成功", "filename": safe_name, "chars": len(text), "chunks": chunk_count}
    except Exception as e:
        logger.error("文档处理失败: %s", e)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="文档处理失败，请稍后重试")


def _build_messages(question: str, context: str, conversation_id: str | None) -> list[dict]:
    """构建 LLM 消息列表（system + history + current question）"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的知识库问答助手。请根据以下检索到的文档片段回答用户问题。"
                "如果文档中没有相关信息，请直接说明。回答时标注信息来源文档。"
                "如果用户的问题是对之前对话的追问或补充，请结合对话历史回答。"
            ),
        },
    ]

    # 添加对话历史
    if conversation_id:
        history = _get_history(conversation_id)
        messages.extend(history)

    # 当前问题（带检索上下文）
    messages.append({
        "role": "user",
        "content": f"检索到的文档片段：\n{context}\n\n问题：{question}",
    })

    return messages


@app.post("/query")
async def ask_question(request: QuestionRequest):
    """非流式查询（兼容旧接口）"""
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")
    if not documents:
        return AnswerResponse(
            answer="还没有上传文档，请先在左侧上传。",
            sources=[],
            conversation_id=request.conversation_id or str(uuid.uuid4()),
        )

    conv_id = request.conversation_id or str(uuid.uuid4())

    # RAG检索
    hits = await asyncio.to_thread(search, request.question)
    if not hits:
        answer = "未找到相关信息，请尝试换个问法。"
        _append_history(conv_id, "user", request.question)
        _append_history(conv_id, "assistant", answer)
        return AnswerResponse(answer=answer, sources=[], conversation_id=conv_id)

    # 拼接检索到的内容作为context
    context = "\n\n---\n\n".join(
        f"【文档: {h['filename']}】\n{h['text']}" for h in hits
    )
    source_files = list(dict.fromkeys(h["filename"] for h in hits))

    # 构建消息（含历史）
    messages = _build_messages(request.question, context, conv_id)

    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=LLM_MODEL,
        messages=messages,
        max_tokens=2000,
    )

    answer = response.choices[0].message.content or ""

    # 记录到对话历史
    _append_history(conv_id, "user", request.question)
    _append_history(conv_id, "assistant", answer)

    return AnswerResponse(
        answer=answer,
        sources=source_files,
        conversation_id=conv_id,
    )


@app.post("/query/stream")
async def ask_question_stream(request: QuestionRequest):
    """SSE 流式查询，逐 token 返回"""
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")

    conv_id = request.conversation_id or str(uuid.uuid4())

    if not documents:
        async def empty_gen():
            yield f"data: {json.dumps({'type': 'answer', 'content': '还没有上传文档，请先在左侧上传。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    async def event_generator():
        # RAG检索（在 generator 内部，失败时返回 SSE error 而非 500）
        try:
            hits = await asyncio.to_thread(search, request.question)
        except Exception as e:
            logger.error("RAG 检索失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': '检索服务异常，请稍后重试'}, ensure_ascii=False)}\n\n"
            return

        if not hits:
            answer = "未找到相关信息，请尝试换个问法。"
            _append_history(conv_id, "user", request.question)
            _append_history(conv_id, "assistant", answer)
            yield f"data: {json.dumps({'type': 'answer', 'content': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': []}, ensure_ascii=False)}\n\n"
            return

        context = "\n\n---\n\n".join(
            f"【文档: {h['filename']}】\n{h['text']}" for h in hits
        )
        source_files = list(dict.fromkeys(h["filename"] for h in hits))
        messages = _build_messages(request.question, context, conv_id)

        full_answer = []
        token_queue: queue.Queue[str | None] = queue.Queue()
        stream_error: list[Exception | None] = [None]

        try:
            # 在线程中运行流式迭代，将 token 实时放入队列
            def _stream_worker():
                try:
                    stream = llm_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        max_tokens=2000,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            token_queue.put(chunk.choices[0].delta.content)
                except Exception as e:
                    stream_error[0] = e
                finally:
                    # 哨兵值：表示流结束
                    token_queue.put(None)

            # 启动线程
            import threading
            worker = threading.Thread(target=_stream_worker, daemon=True)
            worker.start()

            # 异步从队列读取 token 并 yield
            while True:
                # 在事件循环中轮询队列，避免阻塞
                try:
                    token = await asyncio.get_event_loop().run_in_executor(
                        None, token_queue.get, True, 0.1  # 100ms 超时
                    )
                except queue.Empty:
                    continue

                if token is None:
                    # 流结束
                    if stream_error[0]:
                        raise stream_error[0]
                    break

                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("流式生成失败: %s", e)
            _append_history(conv_id, "user", request.question)
            _append_history(conv_id, "assistant", "[生成失败，请重试]")
            yield f"data: {json.dumps({'type': 'error', 'content': '生成回答时出错，请稍后重试'}, ensure_ascii=False)}\n\n"
            return

        answer = "".join(full_answer)
        _append_history(conv_id, "user", request.question)
        _append_history(conv_id, "assistant", answer)

        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': source_files}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
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

    # 阻塞操作放到线程池执行
    await asyncio.to_thread(delete_document_chunks, safe_name)
    file_path = os.path.join("./documents", safe_name)
    if os.path.exists(file_path):
        await asyncio.to_thread(os.remove, file_path)
    return {"message": "已删除", "filename": safe_name}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """清除对话历史"""
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        return {"message": "对话已清除"}
    raise HTTPException(status_code=404, detail="对话不存在")


@app.post("/conversations")
async def create_conversation():
    """创建新对话，返回 conversation_id"""
    conv_id = str(uuid.uuid4())
    _conversations[conv_id] = []
    return {"conversation_id": conv_id}


if __name__ == "__main__":
    import uvicorn
    # 监听 0.0.0.0 仅用于开发/演示，生产环境应限制为 127.0.0.1 或由反向代理暴露
    uvicorn.run(app, host="0.0.0.0", port=8000)
