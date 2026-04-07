# 将长文档切成小块，便于 Embedding 和检索


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """将文本切成重叠的小块"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks
