# 支持加载 PDF、DOCX、TXT 格式的文档

from pathlib import Path
import fitz  # pymupdf，中文 PDF 提取比 PyPDF2 强很多
import docx

__all__ = ["load_document"]


def load_document(file_path: str) -> str:
    """加载文档，返回纯文本内容"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def load_pdf(file_path: str) -> str:
    # 页与页之间用空行分隔，让 rag.chunk_text 能识别段落边界
    with fitz.open(file_path) as doc:
        return "\n\n".join(page.get_text() for page in doc)


def load_docx(file_path: str) -> str:
    """读取 Word 文件（段落之间用空行分隔，保留段落语义）"""
    doc = docx.Document(file_path)
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def load_txt(file_path: str) -> str:
    """读取 TXT 文件，自动检测编码"""
    for encoding in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # latin-1 可以解码任何字节序列，理论上不会到达这里
    # 保留作为防御性兜底
    raise ValueError("无法识别文件编码")
