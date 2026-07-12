"""共享 pytest fixtures"""
import os
import sys
import tempfile
import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_text():
    """一段中文测试文本，包含多个段落"""
    return (
        "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支。\n"
        "它试图理解智能的本质，并生产出一种新的能以与人类智能相似的方式做出反应的智能机器。\n\n"
        "人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。\n"
        "人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。\n\n"
        "机器学习是人工智能的一个重要分支。它使用算法来解析数据，从中学习。\n"
        "深度学习是机器学习的一个子集，使用多层神经网络来进行特征提取和模式识别。\n\n"
        "大语言模型（LLM）是基于Transformer架构的深度学习模型。\n"
        "GPT、BERT、LLaMA等都是知名的大语言模型。它们在文本生成、问答、翻译等任务中表现出色。"
    )


@pytest.fixture
def sample_short_text():
    """短文本，用于测试边界情况"""
    return "这是一段很短的文本。"


@pytest.fixture
def tmp_dir():
    """临时目录，测试结束后自动清理"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_txt_file(tmp_dir, sample_text):
    """创建临时 TXT 文件"""
    path = os.path.join(tmp_dir, "test_doc.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(sample_text)
    return path


@pytest.fixture
def sample_gbk_file(tmp_dir):
    """创建 GBK 编码的 TXT 文件"""
    path = os.path.join(tmp_dir, "gbk_doc.txt")
    with open(path, "w", encoding="gbk") as f:
        f.write("这是一个GBK编码的文件。\n包含中文内容。")
    return path
