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

## Docker 一键启动

```bash
# 前提：已配置 .env（同上）
docker compose up --build

# 访问 http://localhost:3000
```

`chroma_db/`（向量库）和 `documents/`（上传的文档）通过 volume 持久化，重建容器不丢数据。

## 在服务器 / 云电脑上长期运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd frontend && npm install && npm run build && cd ..

./start.sh     # 启动后端(127.0.0.1:8000) + 前端(3000)，日志写入 logs/
./stop.sh      # 停止
```

- 后端只监听 `127.0.0.1:8000`，浏览器统一访问 `http://localhost:3000`，由 Next.js 的 `/api` 代理转发并在服务端注入 `X-API-Key`
- 若机器上配置了 SOCKS 代理（`ALL_PROXY=socks5://...`），openai/httpx 需要额外装 `pip install socksio`，否则启动时报 `ImportError: Using SOCKS proxy`
- 需要开机自启时，把 `start.sh` 写进 `~/.config/autostart/*.desktop`（Linux 桌面）或 systemd unit


## 工作原理

```
用户提问
   ↓
混合检索
├── 向量检索（语义相似度，低于阈值的片段直接丢弃）
└── BM25 检索（关键词匹配）
   ↓
两路分数归一化后加权融合（BM25_WEIGHT 可调）
   ↓
LLM Re-ranking（相关性重排）
   ↓
Top-N 文档片段 + 对话历史（按字符预算截断）
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
| `API_KEY` | ❌ | 接口鉴权密钥。设置后除 `/health` 外所有接口需携带 `X-API-Key` 请求头；前端部署时配置同名 `API_KEY` 环境变量，由 Next.js 服务端 middleware 注入（不进入浏览器 bundle） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | ❌ | 分块大小与重叠字符数（默认 500 / 50） |
| `TOP_K` / `RERANK_TOP_N` | ❌ | 召回数与重排后保留数（默认 10 / 5） |
| `ENABLE_BM25` / `BM25_WEIGHT` | ❌ | 是否启用关键词检索及融合权重（默认 true / 0.3） |
| `ENABLE_RERANK` | ❌ | 是否启用 LLM 重排（默认 true） |
| `SIMILARITY_THRESHOLD` | ❌ | 向量检索相关性门槛，cosine 距离超过该值丢弃（默认 0.65，0 表示不过滤） |
| `MAX_DOC_SIZE_MB` | ❌ | 单文件大小上限 MB（默认 10） |
| `MAX_CHUNKS_PER_DOC` | ❌ | 单文档最大 chunk 数（默认 2000） |
| `MAX_HISTORY_TURNS` / `MAX_HISTORY_CHARS` | ❌ | 多轮对话保留轮数与历史字符预算（默认 10 / 6000） |
| `DOCUMENTS_DIR` / `CHROMA_DIR` | ❌ | 文档与向量库存储路径（默认 ./documents、./chroma_db） |
