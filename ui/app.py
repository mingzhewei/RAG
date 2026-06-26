"""Streamlit entry point - unified search and QA page."""

from __future__ import annotations

import re
import time
from pathlib import Path

import streamlit as st

from bootstrap import (
    apply_compact_style,
    configure_page,
    get_document_manager,
    get_qa_system,
    get_search_engine,
)
from sensor_vector_db.core.types import SearchResult


configure_page("本地传感器文档智能助手")

# ── 自定义紧凑样式 ──
apply_compact_style()
st.markdown(
    """
    <style>
    /* 隐藏默认的页面导航和底部 */
    [data-testid="stSidebarNav"] {display: none;}
    footer {visibility: hidden;}
    
    /* 回答卡片 - 强制深色文字，适配浅色/深色主题 */
    .answer-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        line-height: 1.75;
        color: #1f2937 !important;
    }
    .answer-card * {
        color: #1f2937 !important;
    }
    .answer-card code {
        background: #e5e7eb;
        color: #1f2937 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .answer-card a {
        color: #4f46e5 !important;
    }
    
    /* 来源卡片 */
    .source-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        background: #fafbfc;
        transition: border-color 0.2s;
        color: #374151 !important;
    }
    .source-card:hover {
        border-color: #818cf8;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.8rem;
        color: #6b7280 !important;
    }
    .source-score {
        background: #eef2ff;
        color: #4f46e5 !important;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .source-content {
        font-size: 0.85rem;
        color: #374151 !important;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .highlight {
        background-color: #fef08a;
        color: #1f2937 !important;
        padding: 0 2px;
        border-radius: 2px;
    }
    
    /* 统计卡片 */
    .stat-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f2937 !important;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #9ca3af !important;
        margin-top: 2px;
    }
    
    /* 对话消息 */
    .chat-message {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        margin: 0.4rem 0;
        max-width: 85%;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .chat-message.user {
        background: #eef2ff;
        margin-left: auto;
        text-align: right;
        color: #1f2937 !important;
    }
    .chat-message.assistant {
        background: #f3f4f6;
        margin-right: auto;
        color: #1f2937 !important;
    }
    
    /* 标题 - 适配主题 */
    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .page-subtitle {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 初始化对话历史 ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ── 侧边栏：系统状态和导航 ──
with st.sidebar:
    st.markdown("### 系统状态")
    manager = get_document_manager()
    stats = manager.stats()
    cols = st.columns(3)
    cols[0].markdown(
        f'<div class="stat-card"><div class="stat-number">{stats["documents"]}</div><div class="stat-label">文档</div></div>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        f'<div class="stat-card"><div class="stat-number">{stats["chunks"]}</div><div class="stat-label">片段</div></div>',
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        f'<div class="stat-card"><div class="stat-number">{stats["vectors"]}</div><div class="stat-label">向量</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 功能导航")
    st.page_link("app.py", label="🏠 智能搜索", icon="🔎")
    st.page_link("pages/1_文档导入.py", label="📥 文档导入")
    st.page_link("pages/4_参数抽取.py", label="📊 参数抽取")
    st.page_link("pages/5_参数对比.py", label="📋 参数对比")
    st.page_link("pages/6_系统管理.py", label="⚙️ 系统管理")

    st.markdown("---")
    st.markdown("### 关于")
    st.caption(
        "基于本地向量数据库的传感器技术文档智能检索与问答系统。"
        "支持 PDF、Word、文本和代码文件。"
    )

    # 高级设置折叠
    with st.expander("🔧 高级设置", expanded=False):
        st.caption("以下设置影响检索行为，一般无需修改。")
        st.session_state.rag_top_k = st.slider(
            "检索片段数",
            min_value=3, max_value=20, value=8,
            help="从文档库中检索多少个相关片段作为回答依据。数量越多覆盖面越广，但可能引入噪声。",
        )
        st.session_state.rag_mode = st.selectbox(
            "检索策略",
            ["hybrid", "semantic", "keyword"],
            index=0,
            format_func=lambda x: {"hybrid": "混合（推荐）", "semantic": "语义", "keyword": "关键词"}[x],
            help="混合检索同时利用语义理解和关键词匹配，效果最好。",
        )


# ── 标题区域 ──
st.markdown(
    """
    <div style="text-align:center; padding:1.5rem 0 0.5rem;">
        <h1 style="font-size:1.8rem; font-weight:700; color:#1f2937; margin-bottom:0.3rem;">
            传感器文档智能助手
        </h1>
        <p style="color:#9ca3af; font-size:0.9rem;">
            输入你的问题，从已导入的传感器技术文档中查找答案
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 搜索区域 ──
search_col1, search_col2 = st.columns([0.85, 0.15])
with search_col1:
    query = st.text_input(
        "搜索框",
        placeholder="例如：MPU-6050 的量程是多少？这款传感器的精度如何？",
        label_visibility="collapsed",
        key="search_input",
    )
with search_col2:
    search_clicked = st.button("搜索", type="primary", use_container_width=True, key="search_button")

# ── 快捷示例 ──
if not st.session_state.chat_history:
    example_cols = st.columns(4)
    examples = [
        "这个传感器的量程是多少？",
        "有哪些传感器型号？",
        "精度参数是什么？",
        "如何校准传感器？",
    ]
    for idx, example in enumerate(examples):
        if example_cols[idx].button(
            example, key=f"example_{idx}", use_container_width=True,
        ):
            st.session_state.search_input = example
            st.rerun()


def highlight_keywords(text: str, query: str) -> str:
    """Highlight query keywords in text using HTML spans."""
    if not query.strip():
        return text
    words = [w for w in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_./%+-]+", query.lower()) if len(w) >= 2]
    if not words:
        return text
    pattern = "|".join(re.escape(w) for w in words)
    try:
        highlighted = re.sub(
            f"({pattern})",
            r'<span class="highlight">\1</span>',
            text,
            flags=re.IGNORECASE,
        )
    except re.error:
        return text
    return highlighted


def render_search_result(result: SearchResult, index: int, query: str) -> None:
    """Render one search result card."""
    page_info = f" · 第{result.page_number}页" if result.page_number else ""
    filename = Path(result.file_path).name if result.file_path else "未知文件"
    model_info = f" · {result.sensor_model}" if result.sensor_model else ""

    score_pct = min(100, max(0, int(result.score * 100)))

    content_preview = result.content[:800]
    if len(result.content) > 800:
        content_preview += "..."

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-header">
                <span><strong>S{index}</strong> · {filename}{page_info}{model_info}</span>
                <span class="source-score">相关度 {score_pct}%</span>
            </div>
            <div class="source-content">{highlight_keywords(content_preview, query)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_message(role: str, content: str) -> None:
    """Render a chat-style message."""
    css_class = "user" if role == "user" else "assistant"
    label = "你" if role == "user" else "助手"
    st.markdown(
        f'<div class="chat-message {css_class}"><strong>{label}:</strong> {content}</div>',
        unsafe_allow_html=True,
    )


# ── 处理搜索 ──
do_search = search_clicked or (query and st.session_state.get("search_input") != st.session_state.get("last_search_input"))

if do_search and query.strip():
    st.session_state.last_search_input = query.strip()
    st.session_state.last_query = query.strip()

    # 获取高级设置
    top_k = st.session_state.get("rag_top_k", 8)
    mode = st.session_state.get("rag_mode", "hybrid")

    # 添加用户消息到历史
    st.session_state.chat_history.append({"role": "user", "content": query.strip()})

    # 执行 RAG 问答
    with st.spinner("正在检索并生成回答..."):
        try:
            payload = get_qa_system().answer(query.strip(), top_k=top_k)
        except Exception as e:
            payload = {
                "answer": f"检索或回答过程中出现错误：{e}",
                "sources": [],
            }

    # 先做纯检索（用于展示检索结果）
    search_results = get_search_engine().search(
        query.strip(), mode=mode, top_k=top_k,
    )
    st.session_state.last_search_results = search_results

    # 添加助手回答到历史
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": payload["answer"],
        "sources": search_results,
    })

# ── 显示对话历史 ──
if st.session_state.chat_history:
    st.markdown("---")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            render_chat_message("user", msg["content"])
        else:
            # 助手回答
            st.markdown(
                f'<div class="answer-card">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # 来源
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📎 查看 {len(sources)} 个参考来源", expanded=False):
                    for idx, result in enumerate(sources, start=1):
                        render_search_result(result, idx, st.session_state.get("last_query", ""))

# ── 最近文档（没有对话时显示） ──
if not st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📄 最近导入的文档")
    documents = manager.list_documents()
    if documents:
        recent = documents[:10]
        cols = st.columns(3)
        for idx, doc in enumerate(recent):
            with cols[idx % 3]:
                file_type_icon = {"pdf": "📕", "docx": "📘", "text": "📄", "code": "💻"}.get(doc.get("file_type", ""), "📎")
                st.caption(f"{file_type_icon} {doc.get('filename', '未知')}")
                if doc.get("sensor_model"):
                    st.caption(f"　型号: {doc['sensor_model']}")
    else:
        st.info("还没有导入任何文档。请前往「文档导入」页面添加传感器技术文档。")
