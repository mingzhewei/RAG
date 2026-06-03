"""Streamlit entry point for the local sensor RAG system."""

from __future__ import annotations

from bootstrap import configure_page, get_document_manager

import streamlit as st


configure_page("本地传感器 RAG")

st.title("本地传感器 RAG")

manager = get_document_manager()
stats = manager.stats()
cols = st.columns(3)
cols[0].metric("Documents", stats["documents"])
cols[1].metric("Chunks", stats["chunks"])
cols[2].metric("Vectors", stats["vectors"])

documents = manager.list_documents()
if documents:
    st.subheader("最近导入")
    st.dataframe(documents[:20], use_container_width=True, hide_index=True)
else:
    st.info("尚未导入文档。")

