import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek V4 API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")

# 智谱Embedding配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
EMBEDDING_MODEL = "embedding-3"

# CORS 配置（逗号分隔的域名列表，* 表示允许所有）
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# RAG配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 每个chunk的字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # chunk之间的重叠字符数
TOP_K = int(os.getenv("TOP_K", "10"))  # 向量检索召回的chunk数量（re-ranking前）
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))  # re-ranking后保留的chunk数量
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"  # 是否启用LLM re-ranking

# 混合检索配置
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))  # BM25关键词检索权重（向量检索权重 = 1 - BM25_WEIGHT）
ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"  # 是否启用BM25混合检索

# 对话配置
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))  # 多轮对话保留的最大历史轮数（一问一答=1轮）

# 文档上传限制
MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB
