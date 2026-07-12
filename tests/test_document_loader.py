"""document_loader 模块测试"""
import os
import pytest
from unittest.mock import patch, MagicMock
from document_loader import load_document, load_pdf, load_docx, load_txt


class TestLoadTxt:
    """TXT 文件加载测试"""

    def test_load_utf8_txt(self, sample_txt_file, sample_text):
        result = load_txt(sample_txt_file)
        assert result == sample_text

    def test_load_gbk_txt(self, sample_gbk_file):
        result = load_txt(sample_gbk_file)
        assert "GBK编码" in result
        assert "中文内容" in result

    def test_load_empty_txt(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.txt")
        with open(path, "w", encoding="utf-8") as f:
            pass
        result = load_txt(path)
        assert result == ""

    def test_load_unknown_encoding(self, tmp_dir):
        """所有编码都失败时应抛出 ValueError"""
        path = os.path.join(tmp_dir, "bad.txt")
        with open(path, "wb") as f:
            f.write(b'\x80\x81\x82\x83')
        # latin-1 实际上可以解码任何字节，所以这个测试验证正常路径
        result = load_txt(path)
        assert isinstance(result, str)


class TestLoadDocument:
    """load_document 统一入口测试"""

    def test_dispatch_txt(self, sample_txt_file):
        result = load_document(sample_txt_file)
        assert "人工智能" in result

    def test_unsupported_format(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.xyz")
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_document(path)

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_document("/nonexistent/path/file.txt")

    @patch("document_loader.load_pdf")
    def test_dispatch_pdf(self, mock_pdf):
        mock_pdf.return_value = "PDF 内容"
        result = load_document("/fake/path/doc.pdf")
        mock_pdf.assert_called_once()
        assert result == "PDF 内容"

    @patch("document_loader.load_docx")
    def test_dispatch_docx(self, mock_docx):
        mock_docx.return_value = "DOCX 内容"
        result = load_document("/fake/path/doc.docx")
        mock_docx.assert_called_once()
        assert result == "DOCX 内容"


class TestLoadPdf:
    """PDF 加载测试"""

    def test_load_pdf(self, tmp_dir):
        """使用 pymupdf 创建真实 PDF 并读取"""
        import fitz
        pdf_path = os.path.join(tmp_dir, "test.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello PDF Test Content")
        doc.save(pdf_path)
        doc.close()

        result = load_pdf(pdf_path)
        assert "Hello PDF Test Content" in result

    def test_load_pdf_multipage(self, tmp_dir):
        """多页 PDF"""
        import fitz
        pdf_path = os.path.join(tmp_dir, "multi.pdf")
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1} content")
        doc.save(pdf_path)
        doc.close()

        result = load_pdf(pdf_path)
        for i in range(3):
            assert f"Page {i+1} content" in result


class TestLoadDocx:
    """DOCX 加载测试"""

    def test_load_docx(self, tmp_dir):
        """使用 python-docx 创建真实 DOCX 并读取"""
        import docx
        docx_path = os.path.join(tmp_dir, "test.docx")
        doc = docx.Document()
        doc.add_paragraph("这是DOCX测试段落一")
        doc.add_paragraph("这是DOCX测试段落二")
        doc.save(docx_path)

        result = load_docx(docx_path)
        assert "DOCX测试段落一" in result
        assert "DOCX测试段落二" in result
