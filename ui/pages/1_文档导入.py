"""Persistent document import and directory synchronization page."""

from __future__ import annotations

from pathlib import Path

from bootstrap import configure_page, get_document_manager, get_import_job_manager

import pandas as pd
import streamlit as st


configure_page("文档导入")

st.title("文档导入")

job_manager = get_import_job_manager()

default_path = st.session_state.get("import_path", str(Path("sensor").resolve()))
path = st.text_input("本地文件或文件夹路径", value=default_path, key="import_path")
cols = st.columns([0.22, 0.78])
start = cols[0].button("开始/继续同步", type="primary", disabled=not bool(path.strip()))
cols[1].caption("同步会识别新增、修改、未变化和已删除文件；任务和文件断点持久保存，重启后可继续。")

if start:
    job_id = job_manager.start_import(path.strip())
    st.session_state["selected_import_job"] = job_id
    st.success(f"任务已启动或恢复：{job_id}")
    st.rerun()

jobs = job_manager.list_jobs(limit=20)
if jobs:
    selected_job_id = st.session_state.get("selected_import_job", jobs[0].id)
    job_options = {
        f"{job.created_at.strftime('%Y-%m-%d %H:%M:%S')} · {job.status} · {Path(job.source_path).name} · {job.id[:8]}": job.id
        for job in jobs
    }
    reverse_options = {value: key for key, value in job_options.items()}
    selected_label = st.selectbox(
        "导入任务",
        list(job_options.keys()),
        index=list(job_options.keys()).index(reverse_options.get(selected_job_id, next(iter(job_options)))),
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

    status_cols = st.columns(4)
    status_cols[0].metric("状态", job.status)
    status_cols[1].metric("总文件", job.total_files)
    status_cols[2].metric("已完成", f"{completed_count}/{job.total_files}")
    status_cols[3].metric("待处理", waiting_count)
    detail_cols = st.columns(4)
    detail_cols[0].metric("阶段", job.phase)
    detail_cols[1].metric("新向量化", vectorized_count)
    detail_cols[2].metric("复用/跳过", job.skipped)
    detail_cols[3].metric("失败", job.failed)
    extra_cols = st.columns(4)
    extra_cols[0].metric("正在处理", processing_count)
    extra_cols[1].metric("删除清理", job.deleted)
    extra_cols[2].metric("计划序号", f"{job.current_index}/{job.total_files}")
    extra_cols[3].metric("后台线程", "运行中" if job.is_thread_active else "未运行")
    st.progress(job.progress_ratio)
    st.write(job.message or "")
    if job.current_file:
        st.caption(f"当前文件：{job.current_file}")
    st.caption(
        "总文件表示本次目录中识别到的支持文件；已完成包含新增、更新、复用/跳过和失败；"
        "计划序号用于观察扫描/排队阶段，不等同于已向量化数量。"
        "单文件恢复会复用已写入 SQLite 的 chunk checkpoint，并补齐缺失向量。"
    )

    action_cols = st.columns([0.18, 0.18, 0.64])
    if action_cols[0].button("恢复该任务", disabled=not job.can_resume):
        job_manager.resume_job(job.id)
        st.success("恢复指令已发送。")
        st.rerun()
    if action_cols[1].button("停止该任务", disabled=not job.is_thread_active):
        job_manager.request_stop(job.id)
        st.info("停止请求已发送，任务会在下一个安全检查点中断。")
        st.rerun()
    action_cols[2].caption(
        "如果上次关机或服务重启导致任务停在 running，但后台线程未运行，可点击恢复；"
        "退出早于文件 checkpoint 时会重新处理当前文件。"
    )

    st.subheader("最近事件")
    events = job_manager.get_events(job.id, limit=120)
    if events:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True, height=260)
    else:
        st.info("暂无事件。")

    st.subheader("文件明细")
    file_rows = job_manager.get_file_rows(job.id, limit=1000)
    if file_rows:
        st.caption(f"当前显示 {len(file_rows)} 条文件记录；状态统计：{_status_summary_text(status_counts)}")
        status_filter = st.multiselect(
            "状态过滤",
            sorted({row["status"] for row in file_rows}),
            default=[],
        )
        filtered = [
            row for row in file_rows if not status_filter or row["status"] in status_filter
        ]
        st.dataframe(pd.DataFrame(filtered), width="stretch", hide_index=True, height=320)
    else:
        st.info("任务尚未生成文件计划。")


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
        st.info("尚未导入文档。")
        return
    rows = [
        {
            "文件名": document["filename"],
            "类型": document["file_type"],
            "型号": document.get("sensor_model") or "",
            "厂商": document.get("manufacturer") or "",
            "状态": document["status"],
            "路径": document["file_path"],
        }
        for document in documents
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)


if job:
    _render_import_job_live(job.id)
else:
    st.info("暂无导入任务。")

_render_imported_documents()
