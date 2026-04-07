import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# Embedding 配置（使用 DeepSeek 的 embedding 服务）
EMBEDDING_MODEL = "deepseek-embed"
EMBEDDING_DIM = 1536  # DeepSeek embedding 维度

# RAG 配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# 向量库配置
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
