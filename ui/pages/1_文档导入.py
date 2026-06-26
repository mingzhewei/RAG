"""Persistent document import and directory synchronization page."""

from __future__ import annotations

from pathlib import Path

from bootstrap import configure_page, get_document_manager, get_import_job_manager
from components.safe_dataframe import render_dataframe

import pandas as pd
import streamlit as st


configure_page("文档导入")

st.title("文档导入")
st.caption("将传感器技术文档（PDF、Word、文本、代码）导入本地向量数据库，供后续检索和问答使用。")

job_manager = get_import_job_manager()

# ── 导入区域 ──
st.subheader("导入文件")
default_path = st.session_state.get("import_path", str(Path("sensor").resolve()))
col1, col2 = st.columns([0.7, 0.3])
with col1:
    path = st.text_input(
        "文件或文件夹路径",
        value=default_path,
        key="import_path",
        placeholder="输入本地文件或文件夹的完整路径...",
    )
with col2:
    st.caption("")  # 对齐
    start = st.button("开始导入", type="primary", disabled=not bool(path.strip()), use_container_width=True)

if start:
    job_id = job_manager.start_import(path.strip())
    st.session_state["selected_import_job"] = job_id
    st.success(f"导入任务已启动：{job_id[:8]}...")
    st.rerun()

st.caption("支持 PDF、Word (.docx)、文本文件 (.txt/.md) 和代码文件。系统会自动识别新增、修改和删除的文件。")


# ── 任务列表 ──
jobs = job_manager.list_jobs(limit=20)
if jobs:
    selected_job_id = st.session_state.get("selected_import_job", jobs[0].id)
    job_options = {
        f"{job.created_at.strftime('%m-%d %H:%M')} · {job.status} · {Path(job.source_path).name}": job.id
        for job in jobs
    }
    reverse_options = {value: key for key, value in job_options.items()}
    default_index = 0
    for i, (label, jid) in enumerate(job_options.items()):
        if jid == selected_job_id:
            default_index = i
            break
    selected_label = st.selectbox(
        "历史任务",
        list(job_options.keys()),
        index=default_index,
    )
    selected_job_id = job_options[selected_label]
    st.session_state["selected_import_job"] = selected_job_id
    job = job_manager.get_job(selected_job_id)
else:
    job = None


def _status_summary_text(status_counts: dict[str, int]) -> str:
    """Build a compact status count summary for the file table."""
    if not status_counts:
        return "暂无文件状态"
    return "，".join(f"{status}={count}" for status, count in sorted(status_counts.items()))


def _render_import_job(job_id: str) -> None:
    """Render one import job with fresh status loaded from SQLite."""
    job = job_manager.get_job(job_id)
    if not job:
        st.info("任务不存在或已被删除。")
        return

    st.subheader("任务状态")
    status_counts = job_manager.get_file_status_counts(job.id)
    processing_count = status_counts.get("processing", 0)
    pending_count = status_counts.get("pending", 0)
    waiting_count = pending_count + processing_count
    vectorized_count = job.imported + job.updated
    completed_count = job.completed_files

    # 进度条
    st.progress(job.progress_ratio, text=f"进度 {int(job.progress_ratio * 100)}%")

    # 状态指标
    status_cols = st.columns(4)
    status_cols[0].metric("任务状态", {"running": "运行中", "completed": "已完成", "interrupted": "已中断",
                                       "failed": "失败", "queued": "排队中", "completed_with_errors": "部分失败"}.get(job.status, job.status))
    status_cols[1].metric("文件总数", job.total_files)
    status_cols[2].metric("已完成", f"{completed_count}/{job.total_files}")
    status_cols[3].metric("待处理", waiting_count)

    detail_cols = st.columns(4)
    detail_cols[0].metric("当前阶段", job.phase)
    detail_cols[1].metric("新增/更新", vectorized_count)
    detail_cols[2].metric("跳过", job.skipped)
    detail_cols[3].metric("失败", job.failed)

    st.caption(job.message or "")
    if job.current_file:
        st.caption(f"当前文件：{job.current_file}")

    # 操作按钮
    action_cols = st.columns([0.18, 0.18, 0.64])
    if action_cols[0].button("恢复任务", disabled=not job.can_resume):
        job_manager.resume_job(job.id)
        st.success("恢复指令已发送。")
        st.rerun()
    if action_cols[1].button("停止任务", disabled=not job.is_thread_active):
        job_manager.request_stop(job.id)
        st.info("停止请求已发送，任务会在当前文件处理完成后中断。")
        st.rerun()

    # 最近事件
    with st.expander("📋 最近事件日志", expanded=False):
        events = job_manager.get_events(job.id, limit=120)
        if events:
            render_dataframe(pd.DataFrame(events), width="stretch", hide_index=True, height=260)
        else:
            st.info("暂无事件。")

    # 文件明细
    with st.expander("📁 文件明细", expanded=False):
        file_rows = job_manager.get_file_rows(job.id, limit=1000)
        if file_rows:
            st.caption(f"共 {len(file_rows)} 条文件记录")
            status_filter = st.multiselect(
                "按状态过滤",
                sorted({row["status"] for row in file_rows}),
                default=[],
                key="status_filter",
            )
            filtered = [
                row for row in file_rows if not status_filter or row["status"] in status_filter
            ]
            render_dataframe(pd.DataFrame(filtered), width="stretch", hide_index=True, height=320)
        else:
            st.info("任务尚未生成文件计划。")

    # 失败详情
    failed_rows = job_manager.get_failed_rows(job.id)
    if failed_rows:
        with st.expander(f"⚠️ 失败文件（{len(failed_rows)} 个）", expanded=bool(failed_rows)):
            for row in failed_rows:
                raw_reason = row.get("error") or row.get("message") or "未知原因"
                st.markdown(
                    f"- **{Path(row['file_path']).name}**：{raw_reason[:200]}"
                )


@st.fragment(run_every="5s")
def _render_import_job_live(job_id: str) -> None:
    """Poll import status without a browser-level page refresh."""
    _render_import_job(job_id)


@st.fragment(run_every="15s")
def _render_imported_documents() -> None:
    """Render imported documents and refresh the table locally."""
    st.subheader("已入库文档")
    documents = get_document_manager().list_documents()
    if not documents:
        st.info("还没有导入任何文档。请在上方输入文件路径并点击「开始导入」。")
        return
    rows = [
        {
            "文件名": document["filename"],
            "类型": document["file_type"],
            "型号": document.get("sensor_model") or "-",
            "厂商": document.get("manufacturer") or "-",
            "状态": document["status"],
        }
        for document in documents
    ]
    st.caption(f"共 {len(rows)} 个文档")
    render_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)


if job:
    _render_import_job_live(job.id)
else:
    st.info("还没有导入任务。请在上方输入文件路径并点击「开始导入」。")

_render_imported_documents()
