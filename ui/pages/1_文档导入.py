"""Document import page."""

from __future__ import annotations

from bootstrap import configure_page, get_document_manager
from components.document_card import render_document

import streamlit as st


configure_page("文档导入")

st.title("文档导入")

path = st.text_input("本地文件或文件夹路径", value="")
start = st.button("开始导入", type="primary", disabled=not bool(path.strip()))

if start:
    manager = get_document_manager()
    progress = st.progress(0)
    status = st.empty()

    def update(current: int, total: int, file_path: str) -> None:
        """Update Streamlit progress during import."""
        progress.progress(current / max(total, 1))
        status.write(f"{current}/{total} · {file_path}")

    with st.spinner("正在解析、向量化并入库..."):
        report = manager.import_path(path.strip(), progress_callback=update)
    st.success(
        f"扫描 {report.scanned}，导入 {report.imported}，更新 {report.updated}，"
        f"跳过 {report.skipped}，失败 {report.failed}"
    )
    if report.errors:
        st.error("存在导入失败文件")
        st.dataframe(
            [{"path": str(item.path), "error": item.error} for item in report.errors],
            use_container_width=True,
            hide_index=True,
        )

st.subheader("已入库文档")
manager = get_document_manager()
for document in manager.list_documents():
    render_document(document)

