# 使用 ChromaDB 存储和检索向量

import chromadb
from chromadb.config import Settings
from .embeddings import Embeddings
from config import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME


class VectorStore:
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIRECTORY):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embeddings = Embeddings()
        self.collection = None

    def create_collection(self, collection_name: str = COLLECTION_NAME):
        """创建或获取知识库集合"""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "知识库文档集合"}
        )

    def add_documents(self, documents: list[str], ids: list[str], metadata: list[dict] = None):
        """添加文档到知识库"""
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]

        embeddings = self.embeddings.embed_documents(documents)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """检索最相关的文档"""
        query_embedding = self.embeddings.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return [
            {
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]

    def delete_collection(self, collection_name: str = COLLECTION_NAME):
        """删除知识库集合"""
        self.client.delete_collection(name=collection_name)

    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        if self.collection is None:
            return 0
        return self.collection.count()
