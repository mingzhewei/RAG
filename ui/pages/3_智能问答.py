"""RAG question answering with streaming output and chat history."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from bootstrap import configure_page, get_qa_system, get_search_engine
from sensor_vector_db.core.types import SearchResult


configure_page("智能问答")

st.title("智能问答")
st.caption("多轮对话式问答，AI 会基于已导入的文档回答你的问题。")

# ── 对话历史 ──
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []

# ── 高级设置 ──
with st.sidebar:
    st.markdown("### 问答设置")
    top_k = st.slider(
        "每次参考片段数",
        min_value=3, max_value=20, value=8,
        help="每次问答从文档库中检索多少相关片段。",
    )
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.qa_messages = []
        st.rerun()

# ── 显示历史对话 ──
for msg in st.session_state.qa_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"参考来源（{len(msg['sources'])} 个片段）"):
                for idx, src in enumerate(msg["sources"], start=1):
                    filename = Path(src.file_path).name if src.file_path else "未知"
                    page = f" · p.{src.page_number}" if src.page_number else ""
                    st.caption(f"S{idx} · {filename}{page} · 相关度 {src.score:.0%}")
                    st.text(src.content[:600])
                    if len(src.content) > 600:
                        st.caption("...")

# ── 输入区域 ──
if prompt := st.chat_input("输入你的问题..."):
    # 添加用户消息
    st.session_state.qa_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                payload = get_qa_system().answer(prompt.strip(), top_k=top_k)
                answer = payload["answer"]
                sources = payload["sources"]
            except Exception as e:
                answer = f"生成回答时出错：{e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander(f"参考来源（{len(sources)} 个片段）"):
                for idx, src in enumerate(sources, start=1):
                    filename = Path(src.file_path).name if src.file_path else "未知"
                    page = f" · p.{src.page_number}" if src.page_number else ""
                    st.caption(f"S{idx} · {filename}{page} · 相关度 {src.score:.0%}")
                    st.text(src.content[:600])
                    if len(src.content) > 600:
                        st.caption("...")

    # 保存到历史
    st.session_state.qa_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

