"""Document rendering helpers."""

from __future__ import annotations

import streamlit as st


def render_document(document: dict) -> None:
    """Render one imported document row."""
    with st.container(border=True):
        cols = st.columns([0.35, 0.18, 0.18, 0.14, 0.15])
        cols[0].markdown(f"**{document['filename']}**")
        cols[0].caption(document["file_path"])
        cols[1].write(document.get("sensor_model") or "未识别型号")
        cols[2].write(document.get("manufacturer") or "未识别厂商")
        cols[3].write(document["file_type"])
        cols[4].write(document["status"])

