"""System settings and status page."""

from __future__ import annotations

from bootstrap import (
    apply_runtime_api_key,
    clear_resource_caches,
    configure_page,
    get_document_manager,
)
from sensor_vector_db.config.settings import get_settings

import streamlit as st


configure_page("系统管理")

st.title("系统管理")

settings = get_settings()
manager = get_document_manager()
stats = manager.stats()

cols = st.columns(3)
cols[0].metric("Documents", stats["documents"])
cols[1].metric("Chunks", stats["chunks"])
cols[2].metric("Vectors", stats["vectors"])

st.subheader("路径")
st.code(f"SQLite: {settings.sqlite_path}\nChroma: {settings.chroma_path}\nLog: {settings.log_file}")

st.subheader("DeepSeek")
st.write("API Key 状态：", "已配置" if settings.deepseek_api_key else "未配置")
api_key = st.text_input("DeepSeek API Key", type="password")
if st.button("应用到当前会话", disabled=not bool(api_key.strip())):
    apply_runtime_api_key(api_key)
    clear_resource_caches()
    st.success("已应用。")

st.subheader("Embedding / OCR")
st.write(
    {
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "ocr_enabled": settings.ocr_enabled,
        "ocr_lang": settings.ocr_lang,
        "ocr_max_pages_per_file": settings.ocr_max_pages_per_file,
        "ocr_render_scale": settings.ocr_render_scale,
        "native_thread_limit": settings.native_thread_limit,
    }
)

st.subheader("元数据维护")
st.caption(
    "用最新的型号/厂商识别规则重新解析已入库文档的现有分块，只更新元数据，"
    "不重新解析原文件，也不重新计算向量。"
)
overwrite = st.checkbox("覆盖已有型号/厂商（取消则只补全空字段）", value=True)
if st.button("重算已入库型号/厂商"):
    with st.spinner("正在刷新元数据..."):
        result = manager.refresh_sensor_models(overwrite=overwrite)
    st.success(f"已扫描 {result['scanned']} 个文档，更新 {result['updated']} 个。")

