import os
import json
import hmac
import asyncio
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from collections import OrderedDict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from document_loader import load_document
from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, MAX_DOC_SIZE,
    MAX_HISTORY_TURNS, MAX_HISTORY_CHARS, CORS_ORIGINS, API_KEY, DOCUMENTS_DIR,
)
from rag import add_document, search, delete_document_chunks, get_stats as rag_stats, _build_bm25_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

llm_client: OpenAI | None = None
documents: dict[str, str] = {}  # filename -> text
_doc_lock = asyncio.Lock()

# 限流（按客户端 IP）：上传 10次/分，查询 30次/分
limiter = Limiter(key_func=get_remote_address)
# 流式查询并发上限：超出排队等待，避免 LLM 连接被打满
_stream_semaphore = asyncio.Semaphore(10)

# ── 对话历史管理 ──────────────────────────────────────
# conversation_id -> deque of {role, content}，最多保留 MAX_HISTORY_TURNS * 2 条消息
_conversations: OrderedDict[str, list[dict]] = OrderedDict()
MAX_CONVERSATIONS = 100  # 内存中最多保留的会话数（LRU 淘汰）


def _get_history(conversation_id: str) -> list[dict]:
    """获取对话历史（返回副本），并标记为最近使用"""
    if conversation_id in _conversations:
        # 移到末尾（最近使用）
        _conversations.move_to_end(conversation_id)
        return list(_conversations[conversation_id])
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
    messages = _conversations[conversation_id][-max_msgs:]
    # 再按字符预算从最早的开始丢，避免多轮长回答把上下文顶爆
    total = sum(len(m["content"]) for m in messages)
    while len(messages) > 1 and total > MAX_HISTORY_CHARS:
        total -= len(messages.pop(0)["content"])
    _conversations[conversation_id] = messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    if DEEPSEEK_API_KEY:
        llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60)
    else:
        logger.warning("未设置 DEEPSEEK_API_KEY")

    # 恢复流程会解析 PDF、全量分词建索引，全部放到线程里，避免启动期阻塞事件循环
    def _restore() -> None:
        disk_files: set[str] = set()
        if os.path.exists(DOCUMENTS_DIR):
            for fname in os.listdir(DOCUMENTS_DIR):
                fpath = os.path.join(DOCUMENTS_DIR, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in ALLOWED_EXTENSIONS:
                    try:
                        documents[fname] = load_document(fpath)
                        disk_files.add(fname)
                    except Exception as e:
                        logger.warning("恢复文档 %s 失败: %s", fname, e)
        chroma_files = set(rag_stats()["documents"])
        # Chroma 有但磁盘没有 → 清除孤儿 chunks
        for fname in chroma_files - disk_files:
            logger.info("清除孤儿索引: %s", fname)
            delete_document_chunks(fname)
        # 磁盘有但 Chroma 没有 → 重建索引
        for fname in disk_files - chroma_files:
            logger.info("重建缺失索引: %s", fname)
            try:
                add_document(fname, documents[fname])
            except Exception as e:
                logger.warning("重建索引 %s 失败: %s", fname, e)
        # 启动时重建 BM25 索引
        _build_bm25_index()

    try:
        await asyncio.to_thread(_restore)
    except Exception as e:
        logger.error("知识库恢复失败: %s", e)
    yield


app = FastAPI(title="AI 知识库问答系统", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """可选鉴权：配置 API_KEY 后，除 /health 外的接口都需携带 X-API-Key 请求头"""
    if API_KEY and request.url.path != "/health":
        provided = request.headers.get("x-api-key") or ""
        if not hmac.compare_digest(provided, API_KEY):
            return JSONResponse(status_code=401, content={"detail": "未授权：缺少或错误的 API Key"})
    return await call_next(request)


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
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    # 防止路径遍历：只取文件名部分
    safe_name = os.path.basename(file.filename or "")
    if not safe_name:
        raise HTTPException(status_code=400, detail="无效的文件名")
    file_path = os.path.join(DOCUMENTS_DIR, safe_name)

    try:
        # 分块流式写入，超限即断，避免大文件一次性读入内存
        size = 0
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_DOC_SIZE:
                    f.close()
                    os.remove(file_path)
                    raise HTTPException(status_code=400, detail=f"文件过大，最大允许 {MAX_DOC_SIZE // 1024 // 1024}MB")
                # 写盘是同步阻塞调用，放到线程里执行
                await asyncio.to_thread(f.write, chunk)

        try:
            text = await asyncio.to_thread(load_document, file_path)
        except Exception as e:
            logger.error("解析文档失败 %s: %s", safe_name, e)
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="文档解析失败，请确认文件未损坏")

        # 扫描件/图片型 PDF 提不出文字，若不拦截会"上传成功但检索不到任何内容"
        if len(text.strip()) < 20:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"未能从文档中提取到有效文字（仅 {len(text.strip())} 字符），可能是扫描件或图片型 PDF",
            )

        try:
            chunk_count = await asyncio.to_thread(add_document, safe_name, text)
        except ValueError as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("文档处理失败: %s", e)
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail="文档处理失败，请稍后重试")

        async with _doc_lock:
            documents[safe_name] = text
        return {"message": "上传成功", "filename": safe_name, "chars": len(text), "chunks": chunk_count}
    finally:
        await file.close()


def _build_messages(question: str, context: str, conversation_id: str | None) -> list[dict]:
    """构建 LLM 消息列表（system + history + current question）"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的知识库问答助手。请根据以下检索到的文档片段回答用户问题。"
                "如果文档中没有相关信息，请直接说明。回答时标注信息来源文档。"
                "如果用户的问题是对之前对话的追问或补充，请结合对话历史理解指代，但以本次检索到的片段为唯一事实来源；"
                "历史与检索片段冲突时，以检索片段为准，不要复述历史里已被否定的内容。"
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


def _rewrite_query(question: str, conversation_id: str | None) -> str:
    """
    多轮对话检索前，把追问（如"它呢？""第二个怎么样"）改写为可独立检索的完整问题。
    改写失败或无历史时降级为原问题。
    """
    if llm_client is None or not conversation_id:
        return question
    history = _get_history(conversation_id)
    if not history:
        return question
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是查询改写助手。根据对话历史，把用户的最新问题改写为可以独立理解的完整问题，"
                        "用于文档检索。只输出改写后的问题，不要任何解释。若问题本身已完整，原样输出。"
                    ),
                },
                *history[-6:],
                {"role": "user", "content": question},
            ],
            max_tokens=100,
            temperature=0,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception as e:
        logger.warning("查询改写失败，使用原问题: %s", e)
        return question


@app.post("/query")
@limiter.limit("30/minute")
async def ask_question(request: Request, body: QuestionRequest):
    """非流式查询（兼容旧接口）"""
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")
    if not documents:
        return AnswerResponse(
            answer="还没有上传文档，请先在左侧上传。",
            sources=[],
            conversation_id=body.conversation_id or str(uuid.uuid4()),
        )

    conv_id = body.conversation_id or str(uuid.uuid4())

    # RAG检索（多轮对话时先把追问改写为独立问题）
    search_query = await asyncio.to_thread(_rewrite_query, body.question, conv_id)
    hits = await asyncio.to_thread(search, search_query)
    if not hits:
        answer = "未找到相关信息，请尝试换个问法。"
        _append_history(conv_id, "user", body.question)
        _append_history(conv_id, "assistant", answer)
        return AnswerResponse(answer=answer, sources=[], conversation_id=conv_id)

    # 拼接检索到的内容作为context
    context = "\n\n---\n\n".join(
        f"【文档: {h['filename']}】\n{h['text']}" for h in hits
    )
    source_files = list(dict.fromkeys(h["filename"] for h in hits))

    # 构建消息（含历史）
    messages = _build_messages(body.question, context, conv_id)

    response = await asyncio.to_thread(
        llm_client.chat.completions.create,
        model=LLM_MODEL,
        messages=messages,
        max_tokens=2000,
    )

    answer = response.choices[0].message.content or ""

    # 记录到对话历史
    _append_history(conv_id, "user", body.question)
    _append_history(conv_id, "assistant", answer)

    return AnswerResponse(
        answer=answer,
        sources=source_files,
        conversation_id=conv_id,
    )


@app.post("/query/stream")
@limiter.limit("30/minute")
async def ask_question_stream(request: Request, body: QuestionRequest):
    """SSE 流式查询，逐 token 返回"""
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")

    conv_id = body.conversation_id or str(uuid.uuid4())

    if not documents:
        async def empty_gen():
            yield f"data: {json.dumps({'type': 'answer', 'content': '还没有上传文档，请先在左侧上传。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    async def event_generator():
        # RAG检索（在 generator 内部，失败时返回 SSE error 而非 500）
        try:
            search_query = await asyncio.to_thread(_rewrite_query, body.question, conv_id)
            hits = await asyncio.to_thread(search, search_query)
        except Exception as e:
            logger.error("RAG 检索失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': '检索服务异常，请稍后重试'}, ensure_ascii=False)}\n\n"
            return

        if not hits:
            answer = "未找到相关信息，请尝试换个问法。"
            _append_history(conv_id, "user", body.question)
            _append_history(conv_id, "assistant", answer)
            yield f"data: {json.dumps({'type': 'answer', 'content': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': []}, ensure_ascii=False)}\n\n"
            return

        context = "\n\n---\n\n".join(
            f"【文档: {h['filename']}】\n{h['text']}" for h in hits
        )
        source_files = list(dict.fromkeys(h["filename"] for h in hits))
        messages = _build_messages(body.question, context, conv_id)

        full_answer: list[str] = []
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_error: list[Exception | None] = [None]
        cancelled = threading.Event()
        loop = asyncio.get_running_loop()

        def _push(item: str | None) -> None:
            """线程安全地把 token 交给事件循环（调度失败说明循环已关闭，忽略即可）"""
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, item)
            except RuntimeError:
                pass

        try:
            # 在线程中运行流式迭代，token 通过 call_soon_threadsafe 回传，避免 100ms 轮询空转
            def _stream_worker():
                try:
                    stream = llm_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        max_tokens=2000,
                        stream=True,
                    )
                    for chunk in stream:
                        if cancelled.is_set():
                            break
                        if chunk.choices and chunk.choices[0].delta.content:
                            _push(chunk.choices[0].delta.content)
                except Exception as e:
                    stream_error[0] = e
                finally:
                    # 哨兵值：表示流结束
                    _push(None)

            worker = threading.Thread(target=_stream_worker, daemon=True)
            worker.start()

            while True:
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    # 长时间没有新 token：客户端可能已关闭页面，及时止损（别为没人看的内容付费）
                    if await request.is_disconnected():
                        cancelled.set()
                        logger.info("客户端已断开，终止流式生成")
                        return
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
            _append_history(conv_id, "user", body.question)
            _append_history(conv_id, "assistant", "[生成失败，请重试]")
            yield f"data: {json.dumps({'type': 'error', 'content': '生成回答时出错，请稍后重试'}, ensure_ascii=False)}\n\n"
            return

        answer = "".join(full_answer)
        _append_history(conv_id, "user", body.question)
        _append_history(conv_id, "assistant", answer)

        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'sources': source_files}, ensure_ascii=False)}\n\n"

    async def limited_generator():
        # 并发上限：超出 10 个流式请求时排队等待
        async with _stream_semaphore:
            async for item in event_generator():
                yield item

    return StreamingResponse(
        limited_generator(),
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
    rag_info = await asyncio.to_thread(rag_stats)
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

    # 先删索引和文件，都成功后再从内存列表移除：任一步失败都保持三方可恢复的一致状态
    await asyncio.to_thread(delete_document_chunks, safe_name)
    file_path = os.path.join(DOCUMENTS_DIR, safe_name)
    if os.path.exists(file_path):
        await asyncio.to_thread(os.remove, file_path)
    async with _doc_lock:
        documents.pop(safe_name, None)
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
