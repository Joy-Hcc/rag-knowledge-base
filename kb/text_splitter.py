import re


def _split_sentences(text: str) -> list[str]:
    """按句号、换行等自然边界切分为句子列表"""
    raw = re.split(r"(?<=[。！？\n])\s*", text)
    return [s.strip() for s in raw if s.strip()]


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """将文本按句子边界切成重叠的小块，避免截断单词"""
    sentences = _split_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 单句超长时退化为固定长度切分
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                chunks.append(sentence[i:i + chunk_size].strip())
            continue

        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # 下一块用当前块的尾部作为 overlap，保证上下文连贯
            overlap_text = (
                current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap
                else current_chunk
            )
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
