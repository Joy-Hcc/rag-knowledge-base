import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek V4 API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"

# 文档上传限制
MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB
