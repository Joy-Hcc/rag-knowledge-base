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
    ENABLE_RERANK, ENABLE_BM25, BM25_WEIGHT, SIMILARITY_THRESHOLD,
    MAX_CHUNKS_PER_DOC, CHROMA_DIR,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 20  # 智谱API单次最大处理量
# 重叠不能大于等于 chunk 大小，否则按固定长度切分时会原地踏步
_OVERLAP = max(0, min(CHUNK_OVERLAP, CHUNK_SIZE // 2))

# ── Chroma 向量库 ──────────────────────────────────────
_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# ── Embedding 客户端（智谱API）─────────────────────────
_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL, timeout=60)
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
    global _bm25_index
    all_data = _collection.get(include=["documents", "metadatas"])

    # 构建新索引到临时变量
    new_index = BM25Index()

    if not all_data["ids"]:
        # 整体替换为空索引
        _bm25_index = new_index
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

    # 整体替换引用：查询侧读到的永远是一份完整的索引，不会新旧字段混用
    _bm25_index = new_index


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
def _split_fixed(text: str) -> list[str]:
    """按固定长度 + 重叠切分，用于没有标点的超长片段"""
    step = max(CHUNK_SIZE - _OVERLAP, 1)
    pieces = []
    for i in range(0, len(text), step):
        piece = text[i:i + CHUNK_SIZE].strip()
        if piece:
            pieces.append(piece)
        if i + CHUNK_SIZE >= len(text):
            break
    return pieces


def chunk_text(text: str) -> list[str]:
    """
    语义感知分块策略：
    1. 先按段落（双换行）分割
    2. 长段落再按句号/问号/感叹号分割，块之间保留 _OVERLAP 字符重叠
    3. 超长片段（无标点或单句超长）退化为固定长度切分
    4. 短块向后合并到 CHUNK_SIZE 以内，避免碎片化
    5. 兜底：任何块都不超过 CHUNK_SIZE（embedding API 单条输入有 token 上限）
    """
    if not text or not text.strip():
        return []

    # 第一步：按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 第二步：长段落按句子切分，短段落直接成块
    raw_chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            raw_chunks.append(para)
            continue

        sentences = [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*', para) if s.strip()]
        if len(sentences) <= 1:
            raw_chunks.extend(_split_fixed(para))
            continue

        buffer = ""
        for sent in sentences:
            if len(sent) > CHUNK_SIZE:
                # 单句本身就超长（如整段无标点的表格），先按固定长度拆开
                if buffer:
                    raw_chunks.append(buffer)
                    buffer = ""
                raw_chunks.extend(_split_fixed(sent))
                continue
            if buffer and len(buffer) + len(sent) > CHUNK_SIZE:
                raw_chunks.append(buffer)
                buffer = (buffer[-_OVERLAP:] if _OVERLAP else "") + sent
            else:
                buffer = buffer + sent if buffer else sent
        if buffer:
            raw_chunks.append(buffer)

    # 第三步：短块向后合并，尾部小碎片并回前一块
    min_size = CHUNK_SIZE // 3
    merged: list[str] = []
    buffer = ""
    for chunk in raw_chunks:
        if buffer and len(buffer) + len(chunk) + 1 > CHUNK_SIZE:
            merged.append(buffer)
            buffer = chunk
        else:
            buffer = buffer + "\n" + chunk if buffer else chunk
    if buffer:
        if merged and len(buffer) < min_size and len(merged[-1]) + len(buffer) + 1 <= CHUNK_SIZE * 1.2:
            merged[-1] = merged[-1] + "\n" + buffer
        else:
            merged.append(buffer)

    # 第四步：兜底，确保没有超过 CHUNK_SIZE 的块
    result: list[str] = []
    for chunk in merged:
        if len(chunk) <= CHUNK_SIZE:
            result.append(chunk)
        else:
            result.extend(_split_fixed(chunk))
    return result


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
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        raise ValueError(
            f"文档切分后共 {len(chunks)} 个 chunk，超过单文档上限 {MAX_CHUNKS_PER_DOC}，"
            "请拆分后再上传"
        )

    # 先生成全部 embedding，成功后再替换旧数据，避免 API 失败导致旧 chunks 被删、新数据未入库
    client = _get_embed_client()
    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        embeddings.extend(item.embedding for item in response.data)

    # 删除旧的同名文档chunks，避免重复ID报错
    _collection.delete(where={"filename": filename})

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
    """
    加权分数融合：两路各自归一化到 [0,1] 后按 (1-BM25_WEIGHT) / BM25_WEIGHT 加权求和。

    早期版本用 RRF 名次加权，但 k=60 时名次带来的分差（1/61 与 1/70）远小于权重比，
    权重大的那一路前 N 条会整体压过另一路的第一名 —— 库内 chunk ≥ TOP_K 时
    BM25 结果永远进不了最终 Top-K，混合检索退化成纯向量。
    """
    def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [1.0] * len(values)
        if higher_is_better:
            return [(v - lo) / (hi - lo) for v in values]
        return [(hi - v) / (hi - lo) for v in values]

    fused: dict[str, dict] = {}
    vector_scores = _normalize([h.get("distance", 0.0) for h in vector_hits], higher_is_better=False)
    bm25_scores = _normalize([h.get("score", 0.0) for h in bm25_hits], higher_is_better=True)

    for h, norm in zip(vector_hits, vector_scores):
        key = h["text"] + "::" + h["filename"]
        entry = fused.setdefault(key, {"text": h["text"], "filename": h["filename"], "_score": 0.0})
        entry["_score"] += (1 - BM25_WEIGHT) * norm

    for h, norm in zip(bm25_hits, bm25_scores):
        key = h["text"] + "::" + h["filename"]
        entry = fused.setdefault(key, {"text": h["text"], "filename": h["filename"], "_score": 0.0})
        entry["_score"] += BM25_WEIGHT * norm

    merged = list(fused.values())
    merged.sort(key=lambda x: x["_score"], reverse=True)
    return merged[:TOP_K]


def _vector_search(query: str) -> list[dict]:
    """纯向量检索，低于相关性门槛的 chunk 直接丢弃"""
    # 空集合保护
    count = _collection.count()
    if count == 0:
        return []

    query_embedding = get_embedding(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(TOP_K, count)),
    )

    hits = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        if SIMILARITY_THRESHOLD > 0 and distance > SIMILARITY_THRESHOLD:
            continue
        hits.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            "distance": distance,
        })
    return hits


# ── LLM Re-ranking ────────────────────────────────────
_rerank_client: OpenAI | None = None


def _get_rerank_client() -> OpenAI:
    """复用同一个 LLM 客户端（含连接池）"""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60)
    return _rerank_client


def rerank(query: str, hits: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    LLM Re-ranking：让大模型对检索结果按相关性重新排序。
    使用轻量 prompt，只返回排序后的索引。
    """
    if not hits or len(hits) <= 1:
        return hits[:top_n]

    # 构建候选列表（截断太长会丢掉关键词，500 字符对 10 条候选约 4k token，可接受）
    candidates = []
    for i, h in enumerate(hits):
        snippet = h["text"][:500].replace("\n", " ")
        candidates.append(f"[{i}] 文件 {h['filename']}：{snippet}")
    candidates_text = "\n".join(candidates)

    prompt = (
        f"你是一个文档相关性评估专家。请根据以下用户问题，对候选文档片段按相关性从高到低排序。\n"
        f"只输出全部 {len(hits)} 个候选的完整排列，格式为用逗号分隔的索引（例如 0 到 {len(hits) - 1} 的一个排列），"
        f"不要任何解释、不要遗漏索引。\n\n"
        f"问题：{query}\n\n"
        f"候选片段：\n{candidates_text}"
    )

    try:
        client = _get_rerank_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        order_text = response.choices[0].message.content or ""
        # 解析索引
        indices = [int(x) for x in re.findall(r'\d+', order_text) if int(x) < len(hits)]
        # 去重，保留未出现索引的原始顺序
        seen = set()
        ordered = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                ordered.append(hits[idx])
        # 模型漏掉的候选按原顺序补在后面，但不能声称"重排成功"
        missing = len(hits) - len(seen)
        for i, h in enumerate(hits):
            if i not in seen:
                ordered.append(h)
        if missing:
            logger.warning("Re-ranking 返回不完整：%s，缺失 %d 条", order_text[:50], missing)
        return ordered[:top_n]
    except Exception as e:
        logger.warning("Re-ranking 失败，使用原始排序: %s", e)
        return hits[:top_n]


def search(query: str) -> list[dict]:
    """
    混合检索入口：
    1. 向量检索（带相似度门槛）+ BM25 检索（如启用）
    2. 两路分数归一化后加权融合
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
        h.pop("_score", None)
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
