"""Streamlit UI — CrewAI Agentic RAG 聊天界面。

功能：
- 侧边栏：文档上传 + 向量库状态 + 参数配置
- 主区域：聊天对话界面，支持多轮对话
- 回答展示：带引用高亮 + 路由/性能指标
- Agent 思考过程可视化

启动: uv run streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.chunker import split_documents
from src.document_loader import Document, load_file
from src.vector_store import VectorStore


# ─── 页面配置 ───────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── 会话状态初始化 ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "crew" not in st.session_state:
    st.session_state.crew = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()


def _get_crew():
    """延迟初始化 RAGCrew（避免页面刷新时重复创建）。"""
    if st.session_state.crew is None:
        from src.crew import RAGCrew
        st.session_state.crew = RAGCrew(
            verbose=st.session_state.get("verbose", False),
            memory=st.session_state.get("memory", True),
        )
    return st.session_state.crew


# ─── 侧边栏 ────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 文档管理")

    # 文档上传
    uploaded_files = st.file_uploader(
        "上传文档（PDF / TXT / Markdown）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 灌入向量库", type="primary"):
        store = st.session_state.vector_store
        total_chunks = 0

        with st.spinner("正在处理文档..."):
            for uploaded in uploaded_files:
                # 写入临时文件
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                try:
                    docs = load_file(tmp_path)
                    # 修正 source 元数据为原始文件名
                    for doc in docs:
                        doc.metadata["source"] = uploaded.name
                    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
                    count = store.add_documents(chunks)
                    total_chunks += count
                    st.success(f"✅ {uploaded.name}: {count} 个切片")
                except Exception as e:
                    st.error(f"❌ {uploaded.name}: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        if total_chunks > 0:
            st.info(f"共写入 {total_chunks} 个切片，向量库总计 {store.count} 条")

    # 向量库状态
    st.divider()
    st.subheader("💾 向量库状态")
    store = st.session_state.vector_store
    st.metric("文档切片数", store.count)

    if store.count > 0 and st.button("🗑️ 清空向量库"):
        store.delete_collection()
        st.session_state.vector_store = VectorStore()
        st.rerun()

    # 参数配置
    st.divider()
    st.subheader("⚙️ 参数")
    st.session_state.verbose = st.checkbox("显示 Agent 思考过程", value=False)
    st.session_state.memory = st.checkbox("启用多轮记忆", value=True)

    if st.button("🔄 重置 Crew"):
        st.session_state.crew = None
        st.success("Crew 已重置")

    # 清空对话
    st.divider()
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.session_state.crew = None
        st.rerun()


# ─── 主区域：聊天界面 ──────────────────────────────────────

st.title("🤖 Agentic RAG 知识问答")
st.caption("基于 CrewAI 的多 Agent 协作检索增强生成系统")

# 历史消息渲染
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 展示元信息（仅 assistant 消息）
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            cols = st.columns(3)
            with cols[0]:
                route_emoji = "🔍" if meta.get("route") == "RETRIEVE" else "💬"
                st.caption(f"{route_emoji} 路由: {meta.get('route', '-')}")
            with cols[1]:
                st.caption(f"⏱️ 耗时: {meta.get('elapsed', '-')}s")
            with cols[2]:
                tokens = meta.get("tokens", {})
                if tokens:
                    st.caption(f"🔢 Token: {tokens}")

# 聊天输入
if question := st.chat_input("输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("Agent 协作中..."):
            try:
                crew = _get_crew()
                result = crew.query(question)

                # 渲染回答
                st.markdown(result.answer)

                # 元信息
                meta = {
                    "route": result.route,
                    "elapsed": result.elapsed_seconds,
                    "tokens": result.token_usage,
                }
                cols = st.columns(3)
                with cols[0]:
                    route_emoji = "🔍" if result.route == "RETRIEVE" else "💬"
                    st.caption(f"{route_emoji} 路由: {result.route}")
                with cols[1]:
                    st.caption(f"⏱️ 耗时: {result.elapsed_seconds}s")
                with cols[2]:
                    if result.token_usage:
                        st.caption(f"🔢 Token: {result.token_usage}")

                # Agent 思考过程可视化
                if st.session_state.get("verbose") and result.raw_output:
                    with st.expander("🧠 Agent 思考过程", expanded=False):
                        raw = result.raw_output
                        # 尝试提取 tasks_output
                        if hasattr(raw, "tasks_output"):
                            for i, task_out in enumerate(raw.tasks_output):
                                agent_name = (
                                    task_out.agent
                                    if hasattr(task_out, "agent")
                                    else f"Agent {i+1}"
                                )
                                st.markdown(f"**{agent_name}**")
                                st.code(str(task_out), language=None)
                        else:
                            st.code(str(raw), language=None)

                # 保存消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "metadata": meta,
                })

            except Exception as e:
                error_msg = f"查询出错: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ {error_msg}",
                })
