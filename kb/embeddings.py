# 使用 DeepSeek Embedding API 生成文本向量

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, EMBEDDING_DIM


class Embeddings:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.dimension = EMBEDDING_DIM

    def embed_query(self, text: str) -> list[float]:
        """对单个文本（问题）生成向量"""
        response = self.client.embeddings.create(
            model="deepseek-embed",
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对多个文本（文档块）生成向量"""
        response = self.client.embeddings.create(
            model="deepseek-embed",
            input=texts
        )
        return [item.embedding for item in response.data]
