"""Search result rendering."""

from __future__ import annotations

import streamlit as st

from sensor_vector_db.core.types import SearchResult


def render_result(result: SearchResult, index: int) -> None:
    """Render one search result."""
    page = f" p.{result.page_number}" if result.page_number else ""
    with st.container(border=True):
        cols = st.columns([0.12, 0.68, 0.2])
        cols[0].metric("Score", f"{result.score:.3f}")
        cols[1].markdown(f"**S{index} · {result.source}{page}**")
        cols[1].caption(result.file_path)
        cols[2].caption(result.file_type)
        st.write(result.content[:1600])

