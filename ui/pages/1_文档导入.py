"""Persistent document import and directory synchronization page."""

from __future__ import annotations

from pathlib import Path

from bootstrap import configure_page, get_document_manager, get_import_job_manager
from components.document_card import render_document

import pandas as pd
import streamlit as st


configure_page("文档导入")

st.title("文档导入")

job_manager = get_import_job_manager()

default_path = st.session_state.get("import_path", str(Path("sensor").resolve()))
path = st.text_input("本地文件或文件夹路径", value=default_path, key="import_path")
cols = st.columns([0.22, 0.18, 0.6])
start = cols[0].button("开始/继续同步", type="primary", disabled=not bool(path.strip()))
manual_refresh = cols[1].button("刷新状态")
cols[2].caption("同步会识别新增、修改、未变化和已删除文件；任务状态持久保存，重启后可继续。")

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

if job:
    if job.status == "running" and job.is_thread_active:
        st.markdown("<meta http-equiv='refresh' content='5'>", unsafe_allow_html=True)

    st.subheader("任务状态")
    status_cols = st.columns(6)
    status_cols[0].metric("状态", job.status)
    status_cols[1].metric("阶段", job.phase)
    status_cols[2].metric("文件", f"{job.current_index}/{job.total_files}")
    status_cols[3].metric("新增", job.imported)
    status_cols[4].metric("更新", job.updated)
    status_cols[5].metric("失败", job.failed)
    more_cols = st.columns(3)
    more_cols[0].metric("跳过", job.skipped)
    more_cols[1].metric("删除清理", job.deleted)
    more_cols[2].metric("后台线程", "运行中" if job.is_thread_active else "未运行")
    st.progress(job.progress_ratio)
    st.write(job.message or "")
    if job.current_file:
        st.caption(f"当前文件：{job.current_file}")

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
        "如果上次关机或服务重启导致任务停在 running，但后台线程未运行，可点击恢复。"
    )

    st.subheader("最近事件")
    events = job_manager.get_events(job.id, limit=120)
    if events:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
    else:
        st.info("暂无事件。")

    st.subheader("文件明细")
    file_rows = job_manager.get_file_rows(job.id, limit=1000)
    if file_rows:
        status_filter = st.multiselect(
            "状态过滤",
            sorted({row["status"] for row in file_rows}),
            default=[],
        )
        filtered = [
            row for row in file_rows if not status_filter or row["status"] in status_filter
        ]
        st.dataframe(pd.DataFrame(filtered), width="stretch", hide_index=True)
    else:
        st.info("任务尚未生成文件计划。")
elif manual_refresh:
    st.info("暂无导入任务。")

st.subheader("已入库文档")
manager = get_document_manager()
for document in manager.list_documents():
    render_document(document)
