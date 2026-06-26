"""System settings and status page - simplified for non-technical users."""

from __future__ import annotations

from bootstrap import (
    apply_runtime_llm_settings,
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

# ── 系统概览 ──
st.subheader("系统概览")
cols = st.columns(4)
cols[0].metric("已导入文档", stats["documents"])
cols[1].metric("文档片段", stats["chunks"])
cols[2].metric("向量索引", stats["vectors"])
cols[3].metric("向量维度", settings.embedding_dimension)

# ── 存储位置 ──
with st.expander("📁 存储位置", expanded=False):
    st.code(f"文档数据库: {settings.sqlite_path}\n向量索引:  {settings.chroma_path}\n运行日志:  {settings.log_file}")

# ── AI 模型设置 ──
st.subheader("AI 模型设置")
st.caption("配置用于生成回答的大语言模型。")

current_provider_name = {"crs": "CRS", "deepseek": "DeepSeek", "none": "不使用 AI"}.get(settings.llm_provider, settings.llm_provider)
st.info(f"当前使用：**{current_provider_name}** · 模型：`{settings.active_llm_model}`")

with st.expander("⚙️ 修改 AI 模型配置", expanded=False):
    provider = st.selectbox(
        "模型提供商",
        ["crs", "deepseek", "none"],
        index=["crs", "deepseek", "none"].index(settings.llm_provider),
        format_func=lambda x: {"crs": "CRS", "deepseek": "DeepSeek", "none": "不使用 AI（仅检索）"}[x],
    )

    if provider != "none":
        wire_api = st.selectbox(
            "API 协议",
            ["responses", "chat_completions"],
            index=["responses", "chat_completions"].index(settings.wire_api),
            format_func=lambda x: {"responses": "Responses API", "chat_completions": "Chat Completions API"}[x],
        )
        default_base_url = settings.crs_base_url if provider == "crs" else settings.deepseek_base_url
        default_model = settings.crs if provider == "crs" else settings.deepseek_model
        base_url = st.text_input("API 地址", value=default_base_url)
        model = st.text_input("模型名称", value=default_model)
        api_key = st.text_input("API 密钥（留空则不修改）", type="password",
                                placeholder="输入新的 API 密钥...")
    else:
        wire_api = "responses"
        base_url = ""
        model = ""
        api_key = ""

    if st.button("✅ 应用设置", type="primary"):
        apply_runtime_llm_settings(
            provider=provider,
            api_key=api_key if api_key else None,
            base_url=base_url,
            model=model,
            wire_api=wire_api,
        )
        clear_resource_caches()
        st.success("设置已应用，所有页面将使用新的模型配置。")
        st.rerun()

# ── 文档处理设置（只读） ──
st.subheader("文档处理设置")
st.caption("以下是文档导入时使用的处理参数（需修改 .env 文件后重启生效）。")

with st.expander("📄 查看处理参数", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**文本分块**")
        st.write(f"- 每块大小：{settings.chunk_size} tokens")
        st.write(f"- 重叠大小：{settings.chunk_overlap} tokens")
        st.write(f"- 每次检索返回：{settings.search_top_k} 个片段")

        st.markdown("**嵌入模型**")
        st.write(f"- 模型：{settings.embedding_model}")
        st.write(f"- 设备：{settings.embedding_device}")
        st.write(f"- 批量大小：{settings.embedding_batch_size}")

    with col2:
        st.markdown("**OCR 识别**")
        st.write(f"- 启用：{'是' if settings.ocr_enabled else '否'}")
        st.write(f"- 语言：{settings.ocr_lang}")
        st.write(f"- 每文件最多 OCR 页数：{settings.ocr_max_pages_per_file}")

        st.markdown("**检索权重**")
        st.write(f"- 语义权重：{settings.semantic_weight}")
        st.write(f"- 关键词权重：{settings.bm25_weight}")

# ── 元数据维护 ──
st.subheader("元数据维护")
st.caption("用最新的型号/厂商识别规则重新扫描已入库文档，更新元数据标签（不重新解析文件内容）。")

col1, col2 = st.columns([0.3, 0.7])
with col1:
    overwrite = st.checkbox("覆盖已有型号/厂商", value=True,
                            help="勾选后会覆盖已有标签；取消则只补充空白的标签。")
with col2:
    if st.button("🔄 刷新文档元数据"):
        with st.spinner("正在扫描文档元数据..."):
            result = manager.refresh_sensor_models(overwrite=overwrite)
        st.success(f"扫描了 {result['scanned']} 个文档，更新了 {result['updated']} 个。")
        st.rerun()

# ── 快速操作 ──
st.subheader("快速操作")
st.caption("以下操作会影响整个系统，请谨慎使用。")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔄 刷新页面缓存", use_container_width=True):
        clear_resource_caches()
        st.success("缓存已刷新。")
        st.rerun()
with col2:
    st.caption("")  # 占位
with col3:
    st.caption("")  # 占位

