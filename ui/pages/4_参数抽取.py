"""Parameter extraction page."""

from __future__ import annotations

from bootstrap import configure_page, get_document_manager, get_parameter_extractor

import pandas as pd
import streamlit as st


configure_page("参数抽取")

st.title("参数抽取")

documents = get_document_manager().list_documents()
options = {f"{doc['filename']} · {doc['id']}": doc["id"] for doc in documents}
selected = st.selectbox("文档", list(options.keys())) if options else None
use_llm = st.checkbox("使用 DeepSeek 校验字段名", value=True)

if selected and st.button("抽取参数", type="primary"):
    document_id = options[selected]
    with st.spinner("正在抽取参数..."):
        extractor = get_parameter_extractor()
        parameters = extractor.extract_for_document(
            document_id,
            use_llm=use_llm,
        )
    if extractor.last_warning:
        st.warning(extractor.last_warning)
    rows = [
        {
            "字段": item.normalized_name,
            "原始名称": item.name,
            "值": item.value,
            "单位": item.unit,
            "页码": item.page_number,
            "来源": item.source_text,
            "置信度": item.confidence,
        }
        for item in parameters
    ]
    st.success(f"抽取到 {len(rows)} 个证据参数")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if not documents:
    st.info("尚未导入文档。")
