"""RAG question answering page."""

from __future__ import annotations

from bootstrap import configure_page, get_qa_system
from components.result_card import render_result

import streamlit as st


configure_page("智能问答")

st.title("智能问答")

question = st.text_area("问题", height=120)
top_k = st.slider("检索片段数", min_value=3, max_value=15, value=8)

if st.button("生成回答", type="primary", disabled=not question.strip()):
    with st.spinner("正在检索并生成回答..."):
        payload = get_qa_system().answer(question.strip(), top_k=top_k)
    st.markdown(payload["answer"])
    st.subheader("来源")
    for index, result in enumerate(payload["sources"], start=1):
        render_result(result, index)

