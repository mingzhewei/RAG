"""Simplified search page - for users who prefer raw retrieval results."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from bootstrap import configure_page, get_document_manager, get_search_engine
from sensor_vector_db.core.types import SearchResult


configure_page("文档检索")

st.title("文档检索")
st.caption("直接搜索文档内容，查看原始匹配片段。如需 AI 总结，请使用首页的「智能搜索」。")

# ── 搜索栏 ──
query = st.text_input("搜索关键词或问题", placeholder="输入要查找的内容...")

# ── 高级选项折叠 ──
with st.expander("筛选与设置", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        mode = st.selectbox(
            "检索方式",
            ["hybrid", "semantic", "keyword"],
            index=0,
            format_func=lambda x: {"hybrid": "混合检索", "semantic": "语义检索", "keyword": "关键词检索"}[x],
        )
    with col2:
        top_k = st.slider("返回结果数", min_value=3, max_value=30, value=8)
    with col3:
        file_type = st.selectbox("文件类型", ["全部", "pdf", "docx", "text", "code"])

    col4, col5 = st.columns(2)
    with col4:
        documents = get_document_manager().list_documents()
        models = sorted({doc["sensor_model"] for doc in documents if doc.get("sensor_model")})
        sensor_model = st.selectbox("传感器型号", ["全部", *models])
    with col5:
        manufacturers = sorted({doc["manufacturer"] for doc in documents if doc.get("manufacturer")})
        manufacturer = st.selectbox("制造商", ["全部", *manufacturers])

if st.button("搜索", type="primary", disabled=not query.strip()):
    filters = {
        "file_type": file_type if file_type != "全部" else None,
        "sensor_model": sensor_model if sensor_model != "全部" else None,
        "manufacturer": manufacturer if manufacturer != "全部" else None,
    }
    filters = {k: v for k, v in filters.items() if v}

    with st.spinner("正在搜索..."):
        results = get_search_engine().search(query.strip(), mode=mode, top_k=top_k, filters=filters)

    if not results:
        st.warning("未找到匹配的文档内容，请尝试调整关键词或筛选条件。")
    else:
        st.success(f"找到 {len(results)} 个相关结果")

        # 关键词高亮
        def highlight(text: str, q: str) -> str:
            if not q.strip():
                return text
            words = [w for w in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_./%+-]+", q.lower()) if len(w) >= 2]
            if not words:
                return text
            pattern = "|".join(re.escape(w) for w in words)
            try:
                return re.sub(f"({pattern})", r"**\1**", text, flags=re.IGNORECASE)
            except re.error:
                return text

        for idx, result in enumerate(results, start=1):
            with st.container(border=True):
                header_col1, header_col2, header_col3 = st.columns([0.55, 0.25, 0.2])
                filename = Path(result.file_path).name if result.file_path else "未知"
                page = f" · p.{result.page_number}" if result.page_number else ""
                model = f" · {result.sensor_model}" if result.sensor_model else ""
                header_col1.markdown(f"**S{idx}** · {filename}{page}{model}")
                header_col2.caption(f"来源: {result.source}")
                score_pct = min(100, max(0, int(result.score * 100)))
                header_col3.metric("相关度", f"{score_pct}%")

                st.caption(f"路径: {result.file_path}")
                content_display = highlight(result.content[:2000], query)
                st.markdown(content_display)
                if len(result.content) > 2000:
                    st.caption("... (内容过长，已截断)")

