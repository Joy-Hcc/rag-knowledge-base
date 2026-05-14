# AI 知识库问答系统

上传文档（PDF/Word/TXT），AI 通读全文后回答你的问题，标注来源。利用 DeepSeek V4 百万 token 上下文，无需向量检索。

## 技术栈

| 层 | 技术 |
|---|------|
| 前端 | Streamlit |
| 后端 API | FastAPI |
| LLM | DeepSeek V4 Flash (1M context) |
| 文档解析 | PyPDF2 / python-docx |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
echo "DEEPSEEK_API_KEY=你的Key" > .env

# 3. 启动后端
python api.py

# 4. 启动前端（新终端）
streamlit run app.py

# 5. 访问 http://localhost:8501
```

## API

| 接口 | 说明 |
|------|------|
| `POST /upload` | 上传文档 |
| `POST /query` | 提问 |
| `GET /health` | 健康检查 |
| `GET /stats` | 知识库统计 |
| `DELETE /documents/{name}` | 删除文档 |

## 项目结构

```
├── api.py               # FastAPI 后端
├── app.py               # Streamlit 前端
├── config.py            # 配置
├── document_loader.py   # 文档解析（PDF/Word/TXT）
└── requirements.txt
```

## 工作原理

上传的文档全文存储，提问时将所有文档内容作为上下文，DeepSeek V4 直接从完整文档中找出相关信息并生成答案。
