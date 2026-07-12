# AI 知识库问答系统

上传文档（PDF/Word/TXT），基于 RAG 架构智能检索并回答问题，支持多轮对话和流式输出。

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | Next.js 16 + Tailwind CSS |
| 后端 API | FastAPI + SSE 流式输出 |
| 向量数据库 | ChromaDB |
| Embedding | 智谱 AI embedding-3 |
| LLM | DeepSeek V4 Flash |
| 文档解析 | pymupdf / python-docx |
| 混合检索 | 向量检索 + BM25 + LLM Re-ranking |

## 快速开始

```bash
# 1. 安装后端依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 ZHIPU_API_KEY

# 3. 启动后端
python api.py

# 4. 启动前端（新终端）
cd frontend
npm install
npm run dev

# 5. 访问 http://localhost:3000
```

## 工作原理

```
用户提问
   ↓
混合检索
├── 向量检索（语义相似度）
└── BM25 检索（关键词匹配）
   ↓
RRF 融合排序
   ↓
LLM Re-ranking（相关性重排）
   ↓
Top-N 文档片段 + 对话历史
   ↓
LLM 生成回答（流式输出）
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文档 |
| `/query` | POST | 非流式查询 |
| `/query/stream` | POST | SSE 流式查询 |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 知识库统计 |
| `/documents/{name}` | DELETE | 删除文档 |
| `/conversations` | POST | 创建对话 |
| `/conversations/{id}` | DELETE | 清除对话历史 |

## 项目结构

```
├── api.py               # FastAPI 后端
├── rag.py               # RAG 核心：分块、Embedding、混合检索、Re-ranking
├── config.py            # 配置
├── document_loader.py   # 文档解析（PDF/Word/TXT）
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
├── tests/               # 后端测试
└── frontend/            # Next.js 前端
    ├── src/
    │   ├── app/         # 页面
    │   ├── components/  # 组件
    │   └── lib/         # API 客户端
    └── package.json
```

## 测试

```bash
# 后端测试
pytest

# 前端测试
cd frontend
npm test
```

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API Key |
| `ZHIPU_API_KEY` | ✅ | 智谱 AI API Key（用于 Embedding） |
| `DEEPSEEK_BASE_URL` | ❌ | DeepSeek API 地址（默认官方） |
| `LLM_MODEL` | ❌ | LLM 模型名（默认 deepseek-v4-flash） |
