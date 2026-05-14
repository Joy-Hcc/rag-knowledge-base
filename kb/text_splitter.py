import re


def _split_sentences(text: str) -> list[str]:
    """按句号、换行等自然边界切分为句子列表"""
    # 在句号、问号、感叹号、换行处断开，保留分隔符
    raw = re.split(r"(?<=[。！？\n])\s*", text)
    return [s.strip() for s in raw if s.strip()]


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """将文本按句子边界切成重叠的小块，避免截断单词"""
    sentences = _split_sentences(text)
    chunks = []
    current = ""
    overlap_buffer = ""

    for sentence in sentences:
        # 单句超长时按字符硬切
        if len(sentence) > chunk_size:
            # 先存当前积累的块
            if current:
                chunks.append(current.strip())
                overlap_buffer = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                current = ""
            # 超长句按固定长度切
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                sub = sentence[i:i + chunk_size]
                if sub.strip():
                    chunks.append(sub.strip())
            overlap_buffer = ""
            continue

        candidate = (overlap_buffer + " " + sentence).strip() if overlap_buffer else sentence

        if len(current) + len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            overlap_buffer = current[-chunk_overlap:] if len(current) > chunk_overlap else current
            current = sentence
        else:
            current = candidate if not current else current + " " + sentence
            overlap_buffer = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks
