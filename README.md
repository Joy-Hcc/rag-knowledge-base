# RAG 知识库检索系统

基于 RAG（检索增强生成）的知识库问答系统，上传文档后通过语义检索获取相关内容，由 LLM 生成带来源引用的答案。

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | Streamlit |
| 后端 API | FastAPI |
| 向量数据库 | ChromaDB |
| Embedding | DeepSeek Embedding API |
| LLM | DeepSeek Chat API |
| 文档解析 | PyPDF2 / python-docx |

## 核心链路

```
上传文档(PDF/Word/TXT) → 文本提取 → 句子级分块 → 向量嵌入 → ChromaDB存储
                                                              ↓
用户提问 → 问题向量化 → 语义检索(Top-K) → 拼接上下文 → LLM生成答案+来源
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 启动后端
python api.py

# 4. 启动前端（新终端）
streamlit run app.py

# 5. 浏览器访问 http://localhost:8501
```

## API

| 接口 | 说明 |
|------|------|
| `POST /upload` | 上传文档（PDF/DOCX/TXT） |
| `POST /query` | 提问，返回答案+来源 |
| `GET /health` | 健康检查 |
| `GET /stats` | 知识库统计 |

## 项目结构

```
rag-knowledge-base/
├── api.py              # FastAPI 后端
├── app.py              # Streamlit 前端
├── config.py           # 配置
├── kb/
│   ├── document_loader.py   # 文档加载（PDF/Word/TXT）
│   ├── text_splitter.py     # 句子级文本分块
│   ├── embeddings.py        # Embedding 向量化
│   └── vector_store.py      # ChromaDB 向量存储
└── requirements.txt
```
