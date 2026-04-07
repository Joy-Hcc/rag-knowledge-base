# Streamlit 网页界面

import streamlit as st
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI 知识库问答系统",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI 知识库问答系统")


def check_api_status():
    """检查 API 是否可用"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_stats():
    """获取知识库统计"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=2)
        if response.status_code == 200:
            return response.json().get("count", 0)
    except:
        pass
    return 0


# 侧边栏 - 上传文档
with st.sidebar:
    st.header("📁 上传文档")
    uploaded_file = st.file_uploader(
        "选择文档",
        type=["pdf", "docx", "txt"],
        help="支持 PDF、Word、TXT 格式"
    )

    if uploaded_file is not None:
        if st.button("上传到知识库", type="primary"):
            with st.spinner("上传中..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=60)

                    if response.status_code == 200:
                        st.success("上传成功！")
                        result = response.json()
                        st.info(f"文档已分成 {result['chunks']} 个块")
                    else:
                        error = response.json().get("detail", "上传失败")
                        st.error(f"上传失败: {error}")
                except requests.exceptions.Timeout:
                    st.error("上传超时，请检查文档大小或重试")
                except Exception as e:
                    st.error(f"上传失败: {str(e)}")

    st.divider()

    st.header("📊 知识库状态")
    if check_api_status():
        st.success("API: 已连接")
        doc_count = get_stats()
        st.info(f"文档块数量: {doc_count}")
    else:
        st.error("API: 未连接")
        st.caption("请确保已启动: python api.py")

    st.divider()

    st.header("💡 使用提示")
    st.caption("1. 先上传文档")
    st.caption("2. 在下方提问")
    st.caption("3. 获取基于文档的答案")

# 主界面 - 问答
st.header("💬 提问")

question = st.text_input(
    "输入你的问题",
    placeholder="例如：这份文档的核心内容是什么？",
    help="基于已上传的文档回答问题"
)

if st.button("提问", type="primary") and question:
    with st.spinner("思考中..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()

                st.subheader("📝 答案")
                st.write(data["answer"])

                if data["sources"]:
                    with st.expander("📄 参考来源"):
                        for i, source in enumerate(data["sources"], 1):
                            st.markdown(f"**来源 {i}:** `{source['source']}` (块 {source.get('chunk_id', 0)+1})")
                            st.text(source["content"])
            else:
                error = response.json().get("detail", "请求失败")
                st.error(f"提问失败: {error}")

        except requests.exceptions.Timeout:
            st.error("响应超时，LLM 可能正在处理请重试")
        except Exception as e:
            st.error(f"提问失败: {str(e)}")

elif not question and st.button("提问", type="primary"):
    st.warning("请先输入问题")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 使用步骤

    1. **启动后端**（终端 1）：
       ```
       python api.py
       ```

    2. **启动前端**（终端 2）：
       ```
       streamlit run app.py
       ```

    3. **访问**：打开浏览器访问 http://localhost:8501

    4. **上传文档**：在左侧上传 PDF、Word 或 TXT 文件

    5. **等待处理**：系统会自动将文档分块并存储到向量数据库

    6. **提问问题**：输入关于文档的问题

    7. **获取答案**：基于文档内容生成答案和引用来源

    ### 注意事项

    - 首次运行时，Embedding 模型会下载（约 1GB），请耐心等待
    - 建议单文档不超过 50MB
    - 如果回答较慢，请耐心等待（LLM 生成需要时间）
    """)

# 页脚
st.divider()
st.caption("RAG 知识库问答系统 | 基于 DeepSeek + bge-m3 + ChromaDB")
