# RAG核心逻辑：文档切分 + Embedding + 向量检索

import chromadb
from openai import OpenAI
from config import (
    ZHIPU_API_KEY, ZHIPU_BASE_URL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)

EMBED_BATCH_SIZE = 20  # 智谱API单次最大处理量

# Chroma向量库（本地持久化）
_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Embedding客户端（智谱API）
_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
    return _embed_client


def chunk_text(text: str) -> list[str]:
    """按固定长度切分文本，带重叠，过滤空chunk"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_embedding(text: str) -> list[float]:
    """调用智谱embedding API获取向量"""
    client = _get_embed_client()
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def add_document(filename: str, text: str) -> int:
    """将文档切分后存入向量库，返回chunk数量"""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # 删除旧的同名文档chunks，避免重复ID报错
    _collection.delete(where={"filename": filename})

    # 批量生成embedding（分片调用，避免API限制）
    client = _get_embed_client()
    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        embeddings.extend(item.embedding for item in response.data)

    # 存入Chroma
    ids = [f"{filename}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    _collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query: str) -> list[dict]:
    """检索与query最相关的chunk，返回[{text, filename, distance}]"""
    query_embedding = get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "distance": results["distances"][0][i],
        })
    return hits


def delete_document_chunks(filename: str):
    """删除某个文档的所有chunk"""
    _collection.delete(where={"filename": filename})


def get_stats() -> dict:
    """获取向量库统计"""
    count = _collection.count()
    # 只取 metadatas，不加载 documents 内容，减少内存开销
    all_data = _collection.get(include=["metadatas"])
    filenames = {meta["filename"] for meta in all_data["metadatas"]}
    return {
        "chunk_count": count,
        "documents": sorted(filenames),
    }
