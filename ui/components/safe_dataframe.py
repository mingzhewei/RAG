"""Helpers for Streamlit dataframe rendering during app shutdown."""

from __future__ import annotations

from typing import Any

import streamlit as st


INTERPRETER_SHUTDOWN_MESSAGE = "cannot schedule new futures after interpreter shutdown"


def render_dataframe(data: Any, **kwargs: Any) -> bool:
    """Render a dataframe and suppress only Python shutdown races."""
    try:
        st.dataframe(data, **kwargs)
    except RuntimeError as exc:
        if INTERPRETER_SHUTDOWN_MESSAGE in str(exc):
            return False
        raise
    return True
