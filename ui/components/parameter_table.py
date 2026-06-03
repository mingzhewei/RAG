"""Parameter comparison rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sensor_vector_db.core.parameter_extractor import ComparisonCell


def render_comparison(
    table: dict[str, dict[str, ComparisonCell]],
    models: list[str],
) -> None:
    """Render a comparison table in Streamlit."""
    rows = []
    for parameter_name in sorted(table):
        row = {"参数": parameter_name}
        for model in models:
            cell = table[parameter_name].get(model)
            row[model] = (
                f"{cell.value}\n来源: {cell.source}"
                if cell
                else "未找到\n来源: 未在已入库文档中找到依据"
            )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
