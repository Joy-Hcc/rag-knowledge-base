# 使用本地 bge-m3 embedding 模型生成文本向量

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_DIM


class Embeddings:
    def __init__(self):
        # 使用 bge-m3 多语言 embedding 模型
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.dimension = EMBEDDING_DIM

    def embed_query(self, text: str) -> list[float]:
        """对单个文本（问题）生成向量"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对多个文本（文档块）生成向量"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
