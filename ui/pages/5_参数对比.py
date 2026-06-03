"""Parameter comparison page."""

from __future__ import annotations

from bootstrap import configure_page, get_document_manager, get_parameter_comparer
from components.parameter_table import render_comparison

import streamlit as st


configure_page("参数对比")

st.title("参数对比")

documents = get_document_manager().list_documents()
models = sorted({doc["sensor_model"] for doc in documents if doc.get("sensor_model")})
selected_models = st.multiselect("选择 2-5 个型号", models)

if st.button(
    "生成对比",
    type="primary",
    disabled=not (2 <= len(selected_models) <= 5),
):
    comparer = get_parameter_comparer()
    table = comparer.compare_models(selected_models)
    render_comparison(table, selected_models)
    markdown = comparer.to_markdown(table, selected_models)
    csv_text = comparer.to_csv(table, selected_models)
    st.download_button("下载 Markdown", markdown, "sensor_comparison.md", "text/markdown")
    st.download_button("下载 CSV", csv_text, "sensor_comparison.csv", "text/csv")

if not models:
    st.info("没有可对比的型号。请先导入文档并完成参数抽取。")

