# 支持加载 PDF、DOCX、TXT 格式的文档

import os
from pathlib import Path
import fitz  # pymupdf，中文 PDF 提取比 PyPDF2 强很多
import docx


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
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def load_docx(file_path: str) -> str:
    """读取 Word 文件"""
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def load_txt(file_path: str) -> str:
    """读取 TXT 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
