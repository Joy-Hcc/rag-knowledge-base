# RAG核心逻辑：语义分块 + Embedding + 混合检索 + Re-ranking

import re
import math
import logging
from collections import Counter

import chromadb
import jieba
from openai import OpenAI

from config import (
    ZHIPU_API_KEY, ZHIPU_BASE_URL, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, RERANK_TOP_N,
    ENABLE_RERANK, ENABLE_BM25, BM25_WEIGHT,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 20  # 智谱API单次最大处理量

# ── Chroma 向量库 ──────────────────────────────────────
_chroma_client = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# ── Embedding 客户端（智谱API）─────────────────────────
_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
    return _embed_client


# ── BM25 索引 ─────────────────────────────────────────
# 内存 BM25 索引，使用 jieba 中文分词
# 使用单一数据结构存储，确保原子更新
class BM25Index:
    """BM25 索引数据，原子更新"""

    def __init__(self):
        self.docs: dict[str, list[str]] = {}
        self.chunks: list[tuple[str, str, list[str]]] = []
        self.df: Counter = Counter()
        self.avg_dl: float = 0.0
        self.doc_count: int = 0

_bm25_index = BM25Index()
_bm25_params = {"k1": 1.5, "b": 0.75}


def _tokenize(text: str) -> list[str]:
    """jieba 分词，对中文效果更好，英文按空格分词"""
    text = text.lower().strip()
    if not text:
        return []
    # jieba.cut 返回生成器，转为列表
    tokens = list(jieba.cut(text))
    # 过滤空白 token
    return [t for t in tokens if t.strip()]


def _build_bm25_index():
    """重建 BM25 索引（从 Chroma 拉取所有 chunk，并预分词）"""
    all_data = _collection.get(include=["documents", "metadatas"])

    # 构建新索引到临时变量
    new_index = BM25Index()

    if not all_data["ids"]:
        # 原子替换空索引
        _bm25_index.__dict__.update(new_index.__dict__)
        return

    # 按 filename 分组 + 预分词
    docs_by_file: dict[str, list[str]] = {}
    chunks: list[tuple[str, str, list[str]]] = []
    total_tokens = 0
    df: Counter = Counter()

    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        fname = meta["filename"]
        docs_by_file.setdefault(fname, []).append(doc_text)
        tokens = _tokenize(doc_text)
        chunks.append((fname, doc_text, tokens))
        total_tokens += len(tokens)
        for t in set(tokens):
            df[t] += 1

    new_index.docs = docs_by_file
    new_index.chunks = chunks
    new_index.df = df
    new_index.doc_count = len(chunks)
    new_index.avg_dl = total_tokens / max(new_index.doc_count, 1)

    # 原子替换：一次性更新所有字段
    _bm25_index.__dict__.update(new_index.__dict__)


def _bm25_score(query_tokens: list[str], doc_tokens: list[str],
                avg_dl: float, df: Counter, N: int) -> float:
    """计算单个文档的 BM25 分数。N = 总 chunk 数（非文件数）"""
    k1 = _bm25_params["k1"]
    b = _bm25_params["b"]
    dl = len(doc_tokens)
    tf_map = Counter(doc_tokens)
    score = 0.0
    for token in set(query_tokens):
        tf = tf_map.get(token, 0)
        if tf == 0:
            continue
        idf = math.log((N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator
    return score


def bm25_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """BM25 关键词检索，返回 [{text, filename, score}]"""
    # 快照当前索引状态，避免并发修改
    index = _bm25_index
    if not index.chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for fname, text, tokens in index.chunks:
        score = _bm25_score(query_tokens, tokens, index.avg_dl, index.df, index.doc_count)
        if score > 0:
            scored.append({"text": text, "filename": fname, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── 语义感知分块 ────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    语义感知分块策略：
    1. 先按段落（双换行）分割
    2. 长段落再按句号/问号/感叹号分割
    3. 超长片段才退化为固定长度切分
    4. 短段落合并到 CHUNK_SIZE 以内，避免过度碎片化
    """
    if not text or not text.strip():
        return []

    # 第一步：按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 第二步：处理超长段落 + 合并短段落
    raw_chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            raw_chunks.append(para)
        else:
            # 尝试按句子分割
            sentences = re.split(r'(?<=[。！？.!?])\s*', para)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                # 没有句子分隔符，退化为固定长度切分
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    piece = para[i:i + CHUNK_SIZE].strip()
                    if piece:
                        raw_chunks.append(piece)
            else:
                # 合并句子直到接近 CHUNK_SIZE
                buffer = ""
                for sent in sentences:
                    if buffer and len(buffer) + len(sent) > CHUNK_SIZE:
                        raw_chunks.append(buffer)
                        buffer = sent
                    else:
                        buffer = buffer + sent if buffer else sent
                if buffer:
                    raw_chunks.append(buffer)

    # 第三步：合并过短的 chunk（< CHUNK_SIZE 的 30%）
    min_size = CHUNK_SIZE // 3
    merged: list[str] = []
    buffer = ""
    for chunk in raw_chunks:
        if buffer and len(buffer) + len(chunk) + 1 <= CHUNK_SIZE:
            buffer = buffer + "\n" + chunk
        elif len(chunk) < min_size and not buffer:
            buffer = chunk
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)

    return merged


# ── Embedding ──────────────────────────────────────────
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

    # 重建 BM25 索引
    _build_bm25_index()

    logger.info("文档 %s 入库：%d 个 chunks", filename, len(chunks))
    return len(chunks)


def _merge_results(vector_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 融合向量和 BM25 结果"""
    k = 60  # RRF 常数
    fused: dict[str, dict] = {}

    for rank, h in enumerate(vector_hits):
        key = h["text"] + "::" + h["filename"]
        if key not in fused:
            fused[key] = {"text": h["text"], "filename": h["filename"], "_rrf": 0.0}
        fused[key]["_rrf"] += (1 - BM25_WEIGHT) / (k + rank + 1)

    for rank, h in enumerate(bm25_hits):
        key = h["text"] + "::" + h["filename"]
        if key not in fused:
            fused[key] = {"text": h["text"], "filename": h["filename"], "_rrf": 0.0}
        fused[key]["_rrf"] += BM25_WEIGHT / (k + rank + 1)

    merged = list(fused.values())
    merged.sort(key=lambda x: x["_rrf"], reverse=True)
    return merged[:TOP_K]


def _vector_search(query: str) -> list[dict]:
    """纯向量检索"""
    # 空集合保护
    if _collection.count() == 0:
        return []

    query_embedding = get_embedding(query)
    n_results = min(TOP_K, _collection.count())
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "distance": results["distances"][0][i],
        })
    return hits


# ── LLM Re-ranking ────────────────────────────────────
_rerank_client: OpenAI | None = None


def _get_rerank_client() -> OpenAI:
    """复用同一个 LLM 客户端（含连接池）"""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _rerank_client


def rerank(query: str, hits: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    LLM Re-ranking：让大模型对检索结果按相关性重新排序。
    使用轻量 prompt，只返回排序后的索引。
    """
    if not hits or len(hits) <= 1:
        return hits[:top_n]

    # 构建候选列表
    candidates = []
    for i, h in enumerate(hits):
        snippet = h["text"][:200].replace("\n", " ")
        candidates.append(f"[{i}] {snippet}")
    candidates_text = "\n".join(candidates)

    prompt = (
        f"你是一个文档相关性评估专家。请根据以下用户问题，对候选文档片段按相关性从高到低排序。\n"
        f"只返回排序后的索引列表（如 2,0,4,1,3），不要任何解释。\n\n"
        f"问题：{query}\n\n"
        f"候选片段：\n{candidates_text}"
    )

    try:
        client = _get_rerank_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        order_text = response.choices[0].message.content.strip()
        # 解析索引
        indices = [int(x.strip()) for x in re.findall(r'\d+', order_text) if int(x.strip()) < len(hits)]
        # 去重，保留未出现索引的原始顺序
        seen = set()
        ordered = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                ordered.append(hits[idx])
        for i, h in enumerate(hits):
            if i not in seen:
                ordered.append(h)
        return ordered[:top_n]
    except Exception as e:
        logger.warning("Re-ranking 失败，使用原始排序: %s", e)
        return hits[:top_n]


def search(query: str) -> list[dict]:
    """
    混合检索入口：
    1. 向量检索 + BM25 检索（如启用）
    2. RRF 融合排序
    3. LLM Re-ranking（如启用）
    """
    # 向量检索
    vector_hits = _vector_search(query)

    if ENABLE_BM25:
        bm25_hits = bm25_search(query, top_k=TOP_K)
        hits = _merge_results(vector_hits, bm25_hits)
    else:
        hits = vector_hits

    # Re-ranking
    if ENABLE_RERANK and len(hits) > 1:
        hits = rerank(query, hits)
    else:
        hits = hits[:RERANK_TOP_N]

    # 清理内部评分字段
    for h in hits:
        h.pop("_rrf", None)
        h.pop("_norm_score", None)
        h.pop("distance", None)
        h.pop("score", None)

    return hits


def delete_document_chunks(filename: str):
    """删除某个文档的所有chunk"""
    _collection.delete(where={"filename": filename})
    _build_bm25_index()


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
