"""rag 模块扩展测试：覆盖 add_document, search, rerank, get_stats, _vector_search, _build_bm25_index"""
import pytest
from unittest.mock import patch, MagicMock
from collections import Counter
import rag
from rag import chunk_text, _tokenize, bm25_search, _merge_results, _build_bm25_index


class TestChunkText:
    """语义分块测试"""

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self, sample_short_text):
        chunks = chunk_text(sample_short_text)
        assert len(chunks) == 1
        assert chunks[0] == sample_short_text

    def test_paragraph_splitting(self):
        text = "\n\n".join([
            "人工智能是计算机科学的重要分支领域。" * 20,
            "机器学习通过算法从数据中自动提取规律。" * 20,
            "深度学习使用多层神经网络进行特征提取。" * 20,
        ])
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        joined = "".join(chunks)
        assert "人工智能" in joined
        assert "深度学习" in joined

    def test_no_empty_chunks(self, sample_text):
        chunks = chunk_text(sample_text)
        assert all(c.strip() for c in chunks)

    def test_long_single_paragraph(self):
        long_text = "A" * 2000
        chunks = chunk_text(long_text)
        assert len(chunks) >= 2
        total = sum(len(c) for c in chunks)
        assert total >= 2000

    def test_sentence_splitting(self):
        text = "这是一段关于人工智能的详细描述。" * 80
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 800

    def test_merge_short_paragraphs(self):
        text = "短段落一。\n\n短段落二。\n\n短段落三。"
        chunks = chunk_text(text)
        assert len(chunks) <= 2

    def test_fallback_fixed_length(self):
        """无句子分隔符的超长文本退化为固定长度切分"""
        text = "A" * 1500  # 没有句号，无法按句子分割
        chunks = chunk_text(text)
        assert len(chunks) >= 3  # 1500 / 450 ≈ 3-4 chunks

    def test_only_whitespace_paragraphs(self):
        """全空白段落应被过滤"""
        text = "\n\n   \n\n\n\n"
        assert chunk_text(text) == []


class TestTokenize:
    def test_chinese_text(self):
        tokens = _tokenize("人工智能")
        # jieba 分词结果
        assert len(tokens) > 0
        assert "人工智能" in tokens or "人工" in tokens

    def test_english_text(self):
        tokens = _tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_single_char(self):
        assert _tokenize("A") == ["a"]

    def test_empty(self):
        assert _tokenize("") == []
        assert _tokenize("   ") == []


class TestBM25Search:
    def _set_bm25_data(self, data: dict[str, list[str]]):
        """辅助方法：设置 BM25 索引数据（包括预分词）"""
        index = rag.BM25Index()
        chunks = []
        total_tokens = 0
        df = Counter()
        for fname, texts in data.items():
            for text in texts:
                tokens = rag._tokenize(text)
                chunks.append((fname, text, tokens))
                total_tokens += len(tokens)
                for t in set(tokens):
                    df[t] += 1
        index.docs = data
        index.chunks = chunks
        index.df = df
        index.doc_count = len(chunks)
        index.avg_dl = total_tokens / max(len(chunks), 1)
        # 原子替换
        rag._bm25_index.__dict__.update(index.__dict__)

    def test_empty_index(self):
        original = rag._bm25_index.__dict__.copy()
        empty_index = rag.BM25Index()
        rag._bm25_index.__dict__.update(empty_index.__dict__)
        result = bm25_search("测试查询")
        assert result == []
        rag._bm25_index.__dict__.update(original)

    def test_basic_retrieval(self):
        self._set_bm25_data({
            "ai.txt": ["人工智能是计算机科学的一个分支", "今天天气很好"],
            "ml.txt": ["机器学习使用算法从数据中学习"],
        })
        results = bm25_search("人工智能")
        assert len(results) > 0
        assert "人工智能" in results[0]["text"]

    def test_no_match(self):
        self._set_bm25_data({
            "ai.txt": ["人工智能是计算机科学的一个分支"],
        })
        results = bm25_search("完全不相关的内容xyz")
        for r in results:
            assert r["score"] >= 0

    def test_empty_query_tokens(self):
        """空 token 查询"""
        self._set_bm25_data({"a.txt": ["hello world"]})
        result = bm25_search("")
        assert result == []


class TestMergeResults:
    def test_merge_disjoint(self):
        vector = [{"text": "向量结果A", "filename": "a.txt"}]
        bm25 = [{"text": "关键词结果B", "filename": "b.txt"}]
        merged = _merge_results(vector, bm25)
        assert len(merged) == 2

    def test_merge_overlapping(self):
        vector = [{"text": "相同文本应只出现一次", "filename": "a.txt"}]
        bm25 = [{"text": "相同文本应只出现一次", "filename": "a.txt"}]
        merged = _merge_results(vector, bm25)
        assert len(merged) == 1

    def test_vector_rank_higher(self):
        vector = [
            {"text": "向量第一", "filename": "a.txt"},
            {"text": "向量第二", "filename": "b.txt"},
        ]
        bm25 = [
            {"text": "关键词第一", "filename": "c.txt"},
            {"text": "向量第一", "filename": "a.txt"},
        ]
        merged = _merge_results(vector, bm25)
        assert merged[0]["text"] == "向量第一"

    def test_empty_inputs(self):
        assert _merge_results([], []) == []


class TestBuildBM25Index:
    """BM25 索引构建测试"""

    @patch("rag._collection")
    def test_build_empty_collection(self, mock_coll):
        mock_coll.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        _build_bm25_index()
        assert rag._bm25_index.docs == {}

    @patch("rag._collection")
    def test_build_with_data(self, mock_coll):
        mock_coll.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["chunk1", "chunk2"],
            "metadatas": [
                {"filename": "test.txt"},
                {"filename": "test.txt"},
            ],
        }
        _build_bm25_index()
        assert "test.txt" in rag._bm25_index.docs
        assert len(rag._bm25_index.docs["test.txt"]) == 2


class TestVectorSearch:
    """向量检索测试"""

    @patch("rag._collection")
    def test_empty_collection(self, mock_coll):
        mock_coll.count.return_value = 0
        result = rag._vector_search("test query")
        assert result == []

    @patch("rag.get_embedding")
    @patch("rag._collection")
    def test_normal_search(self, mock_coll, mock_embed):
        mock_coll.count.return_value = 5
        mock_embed.return_value = [0.1] * 128
        mock_coll.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "metadatas": [[{"filename": "a.txt"}, {"filename": "b.txt"}]],
            "distances": [[0.1, 0.3]],
        }
        results = rag._vector_search("query")
        assert len(results) == 2
        assert results[0]["filename"] == "a.txt"


class TestAddDocument:
    """文档入库测试"""

    @patch("rag._build_bm25_index")
    @patch("rag._collection")
    @patch("rag._get_embed_client")
    def test_add_document(self, mock_get_client, mock_coll, mock_build):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 128)]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        count = rag.add_document("test.txt", "短文本测试")
        assert count >= 1
        mock_coll.delete.assert_called_once()
        mock_coll.add.assert_called_once()
        mock_build.assert_called_once()

    @patch("rag._get_embed_client")
    def test_add_empty_document(self, mock_get_client):
        count = rag.add_document("empty.txt", "")
        assert count == 0

    @patch("rag._get_embed_client")
    def test_add_whitespace_only(self, mock_get_client):
        count = rag.add_document("ws.txt", "   \n\n   ")
        assert count == 0


class TestSearch:
    """search 入口函数测试"""

    @patch("rag.ENABLE_BM25", False)
    @patch("rag.ENABLE_RERANK", False)
    @patch("rag._vector_search")
    def test_vector_only(self, mock_vs):
        mock_vs.return_value = [{"text": "t", "filename": "f.txt", "distance": 0.1}]
        results = rag.search("query")
        assert len(results) == 1
        # distance 应被清理
        assert "distance" not in results[0]

    @patch("rag.ENABLE_RERANK", False)
    @patch("rag.bm25_search")
    @patch("rag._vector_search")
    def test_hybrid_search(self, mock_vs, mock_bm25):
        mock_vs.return_value = [{"text": "向量结果", "filename": "a.txt"}]
        mock_bm25.return_value = [{"text": "关键词结果", "filename": "b.txt", "score": 5.0}]
        results = rag.search("query")
        assert len(results) >= 1


class TestGetStats:
    @patch("rag._collection")
    def test_stats(self, mock_coll):
        mock_coll.count.return_value = 10
        mock_coll.get.return_value = {
            "metadatas": [
                {"filename": "a.txt"},
                {"filename": "b.txt"},
                {"filename": "a.txt"},
            ]
        }
        stats = rag.get_stats()
        assert stats["chunk_count"] == 10
        assert "a.txt" in stats["documents"]
        assert "b.txt" in stats["documents"]


class TestDeleteDocumentChunks:
    @patch("rag._build_bm25_index")
    @patch("rag._collection")
    def test_delete(self, mock_coll, mock_build):
        rag.delete_document_chunks("test.txt")
        mock_coll.delete.assert_called_once_with(where={"filename": "test.txt"})
        mock_build.assert_called_once()


class TestRerank:
    @patch("rag._get_rerank_client")
    def test_rerank_single_hit(self, mock_get_client):
        """单条结果不需要 rerank"""
        hits = [{"text": "only one", "filename": "a.txt"}]
        result = rag.rerank("query", hits)
        assert len(result) == 1
        mock_get_client.assert_not_called()

    @patch("rag._get_rerank_client")
    def test_rerank_empty(self, mock_get_client):
        result = rag.rerank("query", [])
        assert result == []

    @patch("rag._get_rerank_client")
    def test_rerank_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "1,0"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        hits = [
            {"text": "第二相关", "filename": "a.txt"},
            {"text": "最相关", "filename": "b.txt"},
        ]
        result = rag.rerank("query", hits)
        assert result[0]["text"] == "最相关"

    @patch("rag._get_rerank_client")
    def test_rerank_fallback_on_error(self, mock_get_client):
        """rerank 失败时降级到原始排序"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_get_client.return_value = mock_client

        hits = [
            {"text": "first", "filename": "a.txt"},
            {"text": "second", "filename": "b.txt"},
        ]
        result = rag.rerank("query", hits)
        # 降级：保持原始顺序
        assert result[0]["text"] == "first"
