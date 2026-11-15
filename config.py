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

# 存储路径
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./documents")  # 上传文档目录
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")  # 向量库目录

# RAG配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 每个chunk的字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # chunk之间的重叠字符数
TOP_K = int(os.getenv("TOP_K", "10"))  # 向量检索召回的chunk数量（re-ranking前）
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))  # re-ranking后保留的chunk数量
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"  # 是否启用LLM re-ranking

# 检索相关性门槛：cosine 距离大于该值的 chunk 直接丢弃（0 表示不过滤）
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

# 混合检索配置
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))  # 两路分数归一化后的 BM25 权重（向量权重 = 1 - BM25_WEIGHT）
ENABLE_BM25 = os.getenv("ENABLE_BM25", "true").lower() == "true"  # 是否启用BM25混合检索

# 对话配置
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))  # 多轮对话保留的最大历史轮数（一问一答=1轮）
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "6000"))  # 历史消息总字符预算（超出从最早的一条开始丢弃）

# 文档上传限制
MAX_DOC_SIZE = int(os.getenv("MAX_DOC_SIZE_MB", "10")) * 1024 * 1024  # 单文件上限，默认 10MB
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "2000"))  # 单文档最大 chunk 数（约 100 万字）

# API 鉴权（可选）：设置后除 /health 外的接口都需要 X-API-Key 请求头
API_KEY = os.getenv("API_KEY", "")
