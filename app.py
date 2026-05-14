import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI 知识库问答", page_icon="📚", layout="wide")
st.title("📚 AI 知识库问答系统")

# ── 侧边栏 ──
with st.sidebar:
    st.header("📁 上传文档")
    uploaded_file = st.file_uploader("选择文档", type=["pdf", "docx", "txt"])

    if uploaded_file:
        if st.button("上传到知识库", type="primary"):
            with st.spinner("上传中..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    r = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"上传成功！({data['chars']} 字符)")
                    else:
                        st.error(r.json().get("detail", "上传失败"))
                except Exception as e:
                    st.error(f"上传失败: {e}")

    st.divider()

    # 文档列表
    st.header("📊 知识库")
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.info(f"文档数: {data['document_count']}  |  总字符: {data['total_chars']:,}")
            for doc in data.get("documents", []):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"📄 {doc}")
                with col2:
                    if st.button("🗑", key=doc, help=f"删除 {doc}"):
                        requests.delete(f"{API_URL}/documents/{doc}")
                        st.rerun()
    except:
        st.error("后端未连接")

# ── 主界面 ──
st.header("💬 提问")

question = st.text_input("输入你的问题", placeholder="例如：这份文档讲了什么？")
if st.button("提问", type="primary"):
    if not question:
        st.warning("请输入问题")
    else:
        with st.spinner("AI 思考中..."):
            try:
                r = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120)
                if r.status_code == 200:
                    data = r.json()
                    st.subheader("📝 答案")
                    st.write(data["answer"])
                    if data.get("sources"):
                        st.caption(f"参考文档: {', '.join(data['sources'])}")
                else:
                    st.error(r.json().get("detail", "请求失败"))
            except Exception as e:
                st.error(f"请求失败: {e}")

# ── 使用说明 ──
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 启动方式
    ```bash
    # 终端 1: 启动后端
    python api.py

    # 终端 2: 启动前端
    streamlit run app.py
    ```
    浏览器访问 http://localhost:8501

    ### 工作流程
    1. 上传 PDF / Word / TXT 文档
    2. 输入问题
    3. AI 通读全文后回答，标注来源文档

    基于 DeepSeek V4 100万 token 上下文，可直接阅读完整文档。
    """)

st.divider()
st.caption("DeepSeek V4 Flash | FastAPI + Streamlit")
