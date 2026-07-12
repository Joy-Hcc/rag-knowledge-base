"""api 模块扩展测试：覆盖 upload, stream, 对话管理等"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import api


@pytest.fixture
def client():
    """FastAPI 测试客户端"""
    api.llm_client = MagicMock()
    api.documents = {"test.txt": "测试文档内容，关于人工智能。"}
    api._conversations.clear()
    yield TestClient(api.app)
    api.llm_client = None
    api.documents = {}
    api._conversations.clear()


@pytest.fixture
def clean_client():
    """无文档的测试客户端"""
    api.llm_client = MagicMock()
    api.documents = {}
    api._conversations.clear()
    yield TestClient(api.app)
    api.llm_client = None
    api.documents = {}
    api._conversations.clear()


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["llm_configured"] is True
        assert data["document_count"] == 1

    def test_health_no_llm(self, client):
        api.llm_client = None
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["llm_configured"] is False


class TestStats:
    def test_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 1
        assert "test.txt" in data["documents"]

    def test_stats_empty(self, clean_client):
        resp = clean_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 0


class TestUpload:
    @patch("api.add_document")
    @patch("api.load_document")
    def test_upload_success(self, mock_load, mock_add, client):
        mock_load.return_value = "文档内容"
        mock_add.return_value = 5
        content = b"fake file content"
        resp = client.post(
            "/upload",
            files={"file": ("test_upload.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "上传成功"
        assert data["chunks"] == 5

    def test_upload_unsupported_format(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "不支持" in resp.json()["detail"]

    def test_upload_empty_filename(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("", b"data", "text/plain")},
        )
        assert resp.status_code == 422

    @patch("api.add_document")
    @patch("api.load_document")
    def test_upload_processing_failure(self, mock_load, mock_add, client):
        """文档处理失败时应清理文件并返回500"""
        mock_load.return_value = "text"
        mock_add.side_effect = Exception("embedding failed")
        resp = client.post(
            "/upload",
            files={"file": ("fail.txt", b"data", "text/plain")},
        )
        assert resp.status_code == 500
        assert "文档处理失败" in resp.json()["detail"]


class TestQuery:
    @patch("api.search")
    def test_query_no_documents(self, mock_search, clean_client):
        resp = clean_client.post("/query", json={"question": "什么是AI？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "没有上传文档" in data["answer"]
        assert data["sources"] == []

    @patch("api.search")
    def test_query_no_results(self, mock_search, client):
        mock_search.return_value = []
        resp = client.post("/query", json={"question": "不存在的问题"})
        assert resp.status_code == 200
        data = resp.json()
        assert "未找到" in data["answer"]

    @patch("api.search")
    def test_query_with_results(self, mock_search, client):
        mock_search.return_value = [
            {"text": "人工智能是计算机科学的分支", "filename": "test.txt"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "人工智能是计算机科学的分支。"
        api.llm_client.chat.completions.create.return_value = mock_response

        resp = client.post("/query", json={"question": "什么是AI？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "人工智能" in data["answer"]
        assert "test.txt" in data["sources"]
        assert "conversation_id" in data

    @patch("api.search")
    def test_query_with_conversation_id(self, mock_search, client):
        mock_search.return_value = [
            {"text": "机器学习是AI的子领域", "filename": "test.txt"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "机器学习是AI的子领域。"
        api.llm_client.chat.completions.create.return_value = mock_response

        resp1 = client.post("/query", json={
            "question": "什么是AI？",
            "conversation_id": "test-conv-123"
        })
        assert resp1.status_code == 200
        assert resp1.json()["conversation_id"] == "test-conv-123"

        history = api._get_history("test-conv-123")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @patch("api.search")
    def test_query_no_llm(self, mock_search, client):
        api.llm_client = None
        resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 500

    @patch("api.search")
    def test_query_auto_generates_conversation_id(self, mock_search, client):
        """不传 conversation_id 时应自动生成"""
        mock_search.return_value = [
            {"text": "测试内容", "filename": "test.txt"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "回答"
        api.llm_client.chat.completions.create.return_value = mock_response

        resp = client.post("/query", json={"question": "问题"})
        assert resp.status_code == 200
        conv_id = resp.json()["conversation_id"]
        assert conv_id is not None
        assert len(conv_id) > 0


class TestStreamQuery:
    @patch("api.search")
    def test_stream_no_documents(self, mock_search, clean_client):
        resp = clean_client.post("/query/stream", json={"question": "test"})
        assert resp.status_code == 200
        # 解析 SSE data 事件
        import json
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        assert len(events) >= 1
        # 第一个事件应该是 answer 类型
        assert events[0]["type"] == "answer"
        assert "没有上传文档" in events[0]["content"]

    @patch("api.search")
    def test_stream_no_results(self, mock_search, client):
        mock_search.return_value = []
        resp = client.post("/query/stream", json={"question": "不存在的问题"})
        assert resp.status_code == 200
        import json
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        assert len(events) >= 1
        assert events[0]["type"] == "answer"
        assert "未找到" in events[0]["content"]

    @patch("api.search")
    def test_stream_success(self, mock_search, client):
        mock_search.return_value = [
            {"text": "AI是人工智能", "filename": "test.txt"},
        ]
        # Mock streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "人工"
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "智能"

        api.llm_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

        resp = client.post("/query/stream", json={"question": "什么是AI？"})
        assert resp.status_code == 200
        assert "人工" in resp.text
        assert "智能" in resp.text
        # 应该有 done 事件
        assert '"type": "done"' in resp.text or '"type":"done"' in resp.text

    @patch("api.search")
    def test_stream_no_llm(self, mock_search, client):
        api.llm_client = None
        resp = client.post("/query/stream", json={"question": "test"})
        assert resp.status_code == 500


class TestConversations:
    def test_create_conversation(self, client):
        resp = client.post("/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        # 验证后端确实记录了
        assert data["conversation_id"] in api._conversations

    def test_delete_conversation(self, client):
        api._conversations["test-conv"] = [{"role": "user", "content": "hi"}]
        resp = client.delete("/conversations/test-conv")
        assert resp.status_code == 200
        assert "test-conv" not in api._conversations

    def test_delete_nonexistent_conversation(self, client):
        resp = client.delete("/conversations/nonexistent")
        assert resp.status_code == 404


class TestConversationHistory:
    def test_append_history(self):
        api._conversations.clear()
        api._append_history("conv1", "user", "你好")
        api._append_history("conv1", "assistant", "你好！")
        history = api._get_history("conv1")
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "你好"}

    def test_history_truncation(self):
        api._conversations.clear()
        from config import MAX_HISTORY_TURNS
        for i in range(MAX_HISTORY_TURNS * 2 + 10):
            role = "user" if i % 2 == 0 else "assistant"
            api._append_history("conv2", role, f"消息 {i}")
        history = api._get_history("conv2")
        assert len(history) == MAX_HISTORY_TURNS * 2

    def test_lru_eviction(self):
        api._conversations.clear()
        # 用 OrderedDict 的方式填充
        for i in range(api.MAX_CONVERSATIONS):
            api._conversations[f"conv-{i}"] = [{"role": "user", "content": f"msg {i}"}]
        # 触发淘汰
        api._append_history("new-conv", "user", "新消息")
        # 最早的应被淘汰
        assert "conv-0" not in api._conversations
        assert "new-conv" in api._conversations

    def test_lru_access_refreshes(self):
        """访问一个会话应刷新其 LRU 位置"""
        api._conversations.clear()
        for i in range(api.MAX_CONVERSATIONS):
            api._conversations[f"conv-{i}"] = []
        # 访问最早的会话，使其变为最新
        api._get_history("conv-0")
        # 添加新会话触发淘汰
        api._append_history("new-conv", "user", "新消息")
        # conv-0 因为被访问过，不应被淘汰；conv-1 应该被淘汰
        assert "conv-0" in api._conversations
        assert "conv-1" not in api._conversations

    def test_get_history_nonexistent(self):
        api._conversations.clear()
        assert api._get_history("nonexistent") == []


class TestDeleteDocument:
    @patch("api.delete_document_chunks")
    def test_delete_success(self, mock_delete, client):
        resp = client.delete("/documents/test.txt")
        assert resp.status_code == 200
        assert resp.json()["message"] == "已删除"
        mock_delete.assert_called_once()

    def test_delete_nonexistent(self, client):
        resp = client.delete("/documents/nonexistent.txt")
        assert resp.status_code == 404
