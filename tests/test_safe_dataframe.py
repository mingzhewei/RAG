"""Tests for Streamlit dataframe shutdown handling."""

from __future__ import annotations

import pytest

from ui.components import safe_dataframe


def test_render_dataframe_suppresses_interpreter_shutdown(monkeypatch) -> None:
    """The shutdown race from pyarrow should not surface as a UI traceback."""

    def raise_shutdown(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after interpreter shutdown")

    monkeypatch.setattr(safe_dataframe.st, "dataframe", raise_shutdown)

    assert safe_dataframe.render_dataframe([{"a": 1}]) is False


def test_render_dataframe_reraises_other_runtime_errors(monkeypatch) -> None:
    """Non-shutdown dataframe failures should remain visible."""

    def raise_other(*args, **kwargs):
        raise RuntimeError("real dataframe conversion failure")

    monkeypatch.setattr(safe_dataframe.st, "dataframe", raise_other)

    with pytest.raises(RuntimeError, match="real dataframe conversion failure"):
        safe_dataframe.render_dataframe([{"a": 1}])
