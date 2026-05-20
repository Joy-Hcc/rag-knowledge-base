import gradio as gr
import requests

API_URL = "http://localhost:8000"


def upload_document(file):
    """上传文档到知识库"""
    if file is None:
        return "请选择文件"

    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f)}
            r = requests.post(f"{API_URL}/upload", files=files, timeout=60)

        if r.status_code == 200:
            data = r.json()
            return f"上传成功！({data['chars']} 字符)"
        else:
            return f"上传失败: {r.json().get('detail', '未知错误')}"
    except Exception as e:
        return f"上传失败: {e}"


def ask_question(question, history):
    """向知识库提问"""
    if not question.strip():
        return history, ""

    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=120
        )

        if r.status_code == 200:
            data = r.json()
            answer = data["answer"]
            sources = data.get("sources", [])

            if sources:
                answer += f"\n\n---\n参考文档: {', '.join(sources)}"

            history.append([question, answer])
            return history, ""
        else:
            error = r.json().get("detail", "请求失败")
            history.append([question, f"错误: {error}"])
            return history, ""
    except Exception as e:
        history.append([question, f"请求失败: {e}"])
        return history, ""


def get_stats():
    """获取知识库统计信息"""
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        if r.status_code == 200:
            data = r.json()
            docs = data.get("documents", [])
            doc_list = "\n".join([f"• {doc}" for doc in docs]) if docs else "暂无文档"
            return f"文档数: {data['document_count']}  |  总字符: {data['total_chars']:,}\n\n{doc_list}"
        else:
            return "无法获取统计信息"
    except:
        return "后端未连接"


def delete_document(filename):
    """删除文档"""
    try:
        r = requests.delete(f"{API_URL}/documents/{filename}", timeout=3)
        if r.status_code == 200:
            return f"已删除: {filename}"
        else:
            return f"删除失败: {r.json().get('detail', '未知错误')}"
    except Exception as e:
        return f"删除失败: {e}"


def refresh_stats():
    """刷新统计信息"""
    return get_stats()


# 创建Gradio界面
with gr.Blocks(
    title="AI 知识库问答系统"
) as demo:

    gr.Markdown(
        """
        # 📚 AI 知识库问答系统
        上传文档，AI 通读全文后回答你的问题
        """,
        elem_classes="header"
    )

    with gr.Row():
        # 左侧：文档管理
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文档管理")

            with gr.Group():
                file_input = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".docx", ".txt"],
                    type="filepath"
                )
                upload_btn = gr.Button("上传到知识库", variant="primary")
                upload_output = gr.Textbox(label="上传状态", interactive=False)

            gr.Markdown("### 📊 知识库统计")
            stats_output = gr.Textbox(
                label="统计信息",
                value=get_stats(),
                interactive=False,
                elem_classes="stats-box"
            )
            refresh_btn = gr.Button("刷新统计")

        # 右侧：问答界面
        with gr.Column(scale=2):
            gr.Markdown("### 💬 提问")
            chatbot = gr.Chatbot(
                label="对话记录",
                height=400
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：这份文档讲了什么？",
                    scale=4
                )
                ask_btn = gr.Button("发送", variant="primary", scale=1)

    # 绑定事件
    upload_btn.click(
        fn=upload_document,
        inputs=file_input,
        outputs=upload_output
    ).then(
        fn=refresh_stats,
        outputs=stats_output
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input]
    )

    refresh_btn.click(
        fn=refresh_stats,
        outputs=stats_output
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown(
            """
            ### 启动方式
            ```bash
            # 终端 1: 启动后端
            python api.py

            # 终端 2: 启动前端
            python app_gradio.py
            ```
            浏览器访问 http://localhost:7860

            ### 工作流程
            1. 上传 PDF / Word / TXT 文档
            2. 输入问题
            3. AI 通读全文后回答，标注来源文档

            基于 DeepSeek V4 100万 token 上下文，可直接阅读完整文档。
            """
        )

    gr.Markdown("---\nDeepSeek V4 Flash | FastAPI + Gradio")


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        theme=gr.themes.Soft(),
        css="""
            .container { max-width: 1200px; margin: auto; }
            .header { text-align: center; margin-bottom: 2rem; }
            .stats-box { background: #f0f7ff; padding: 1rem; border-radius: 0.5rem; }
        """
    )