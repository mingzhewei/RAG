"""Search page."""

from __future__ import annotations

from bootstrap import configure_page, get_document_manager, get_search_engine
from components.result_card import render_result

import streamlit as st


configure_page("智能检索")

st.title("智能检索")

query = st.text_input("检索问题或关键词")
cols = st.columns([0.25, 0.25, 0.25, 0.25])
mode = cols[0].selectbox("模式", ["hybrid", "semantic", "keyword"], index=0)
top_k = cols[1].number_input("Top K", min_value=1, max_value=30, value=8)
file_type = cols[2].selectbox("文件类型", ["", "pdf", "docx", "text", "code"])

documents = get_document_manager().list_documents()
models = sorted({doc["sensor_model"] for doc in documents if doc.get("sensor_model")})
sensor_model = cols[3].selectbox("型号", ["", *models])

if st.button("检索", type="primary", disabled=not query.strip()):
    filters = {
        "file_type": file_type or None,
        "sensor_model": sensor_model or None,
    }
    with st.spinner("正在检索..."):
        results = get_search_engine().search(query, mode=mode, top_k=int(top_k), filters=filters)
    if not results:
        st.warning("未在已入库文档中找到依据。")
    for index, result in enumerate(results, start=1):
        render_result(result, index)

