import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from document_loader import load_document
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL

llm_client: OpenAI = None
documents: dict[str, str] = {}  # filename -> text


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    if DEEPSEEK_API_KEY:
        llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    else:
        print("警告: 未设置 DEEPSEEK_API_KEY")
    yield


app = FastAPI(title="AI 知识库问答系统", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    upload_dir = "./documents"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        text = load_document(file_path)
        documents[file.filename] = text
        return {"message": "上传成功", "filename": file.filename, "chars": len(text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档解析失败: {e}")


@app.post("/query", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM 未配置")
    if not documents:
        return AnswerResponse(answer="还没有上传文档，请先在左侧上传。", sources=[])

    context = "\n\n---\n\n".join(
        f"【文档: {name}】\n{text}" for name, text in documents.items()
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个专业的知识库问答助手。请根据以下提供的文档内容回答用户问题。"
                    "如果文档中没有相关信息，请直接说明。回答时标注信息来源文档。"
                ),
            },
            {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{request.question}"},
        ],
        max_tokens=2000,
    )

    return AnswerResponse(
        answer=response.choices[0].message.content,
        sources=list(documents.keys()),
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
    return {
        "document_count": len(documents),
        "total_chars": sum(len(t) for t in documents.values()),
        "documents": list(documents.keys()),
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    if filename not in documents:
        raise HTTPException(status_code=404, detail="文档不存在")
    del documents[filename]
    file_path = os.path.join("./documents", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": "已删除", "filename": filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
