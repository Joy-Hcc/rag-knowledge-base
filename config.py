import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek V4 API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"

# 智谱Embedding配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
EMBEDDING_MODEL = "embedding-3"

# RAG配置
CHUNK_SIZE = 500  # 每个chunk的字符数
CHUNK_OVERLAP = 50  # chunk之间的重叠字符数
TOP_K = 5  # 检索返回的chunk数量

# 文档上传限制
MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB
