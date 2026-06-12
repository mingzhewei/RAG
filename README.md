# 本地传感器 RAG 检索 MVP

这是一个本地运行的传感器技术文档 RAG 检索系统。系统以本地向量库为核心，支持 PDF、DOCX、受控文本和代码文件导入，扫描 PDF 可启用 PaddleOCR，本地 embedding 默认使用 `BAAI/bge-m3`，问答生成通过 DeepSeek Chat API 完成。

## 运行

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python main.py
```

`main.py` 会初始化本地 SQLite/Chroma 目录并自动启动 Streamlit 界面。只检查环境、不启动界面时使用：

```powershell
.\.venv\Scripts\python main.py --check
```

如果默认端口被占用，可以指定端口：

```powershell
.\.venv\Scripts\python main.py --port 8502
```

`DEEPSEEK_API_KEY` 只写入 `.env` 或在 Streamlit 系统管理页临时输入，不要提交到 Git。

## 轻量测试

测试默认使用 deterministic fake embedding，不会下载 BGE-M3，也不会调用 DeepSeek。

```powershell
.\.venv\Scripts\pytest
```

## 准确性原则

- 回答必须基于检索片段。
- 参数值必须保留来源文本、页码或 chunk 信息。
- 未找到依据时显示“未在已入库文档中找到依据”，不推断、不补全。

## 导入范围

- `.txt` 仅导入不大于 1 MiB（1,048,576 bytes）的文件；大于该阈值的 TXT 会被扫描阶段排除。
- CSV 文件一律排除，不作为 RAG 证据文本入库。
- 数据库相关文件一律排除，包括 `.db`、`.sqlite`、`.sqlite3`、`.sqlite-wal`、`.sqlite-shm`、`.sql`、`.ddl`、`.dml`、`.dump`、`.bak`，以及 Chroma 运行库目录内的文件。

## 持久导入

- 文档导入页使用持久任务表记录目录同步进度。
- 同一目录再次同步时会按文件哈希识别新增、修改和未变化文件。
- 源目录中删除的文件会在下一次同步时清理对应 SQLite 元数据和 Chroma 向量。
- 如果关机或服务重启导致任务中断，回到文档导入页选择任务并点击“恢复该任务”即可继续处理未完成或失败文件；已完成文件会跳过。
- 单个文件已完成解析/分块并写入 SQLite checkpoint 后，如果退出发生在向量化或写 Chroma 阶段，下次会只补齐缺失 chunk 向量；如果退出早于这个 checkpoint，则该文件会重新解析处理。
- 文件级明细会显示成功、跳过、失败、当前阶段和错误原因。

### 导入/恢复使用流程

1. 将待入库资料放在本地目录中，例如 `sensor/`。
2. 打开 Streamlit 的“文档导入”页，输入本地文件或文件夹路径，点击“开始/继续同步”。
3. 需要暂停时点击“停止该任务”；任务会在下一个安全检查点退出。
4. 如果浏览器关闭、服务重启或电脑关机，重新启动项目后回到“文档导入”页，选择原任务并点击“恢复该任务”。
5. 查看“文件明细”确认每个文件的状态；`skipped` 表示文件未变化或已复用已有向量，`processing`/`pending` 表示仍需继续。

## 导入加速

- 默认先不做 OCR：`OCR_ENABLED=false`。PDF 会先尽量抽取可复制文本、表格并完成向量化。
- 文字版资料入库完成后，可以把 `.env` 改成 `OCR_ENABLED=true` 再恢复导入；程序会识别 text-only 索引无法满足 OCR 目标，并重建需要 OCR 的文件。
- 重建时会先写入临时 `indexing` 状态和 SQLite chunk checkpoint，只有 SQLite 元数据和 Chroma 向量都写成功后才标记为 `imported`，避免半成品污染检索。
- 恢复 `indexing` 文件时会先检查 Chroma 中已存在的 chunk id，只对缺失 chunk 重新 embedding 并写入向量库。
- 默认每个 PDF 最多 OCR 20 页，防止长扫描件拖垮内存和磁盘；`OCR_MAX_PAGES_PER_FILE=0` 表示不限制。
- `OCR_RENDER_SCALE=1.5` 会降低 OCR 渲染图片尺寸，通常比 `2.0` 更省内存和 SSD 写入。
- BGE embedding 在 CPU 上会很慢。有可用 NVIDIA GPU 时，可尝试 `EMBEDDING_DEVICE=cuda` 和 `EMBEDDING_USE_FP16=true`。
- `EMBEDDING_BATCH_SIZE` 调大可能更快，但会增加内存压力；机器已经卡顿时不要盲目调大。
- `CHUNK_SIZE` 调大可以减少向量数量、加快导入，但检索粒度会变粗。
- 默认 `NATIVE_THREAD_LIMIT=4`，限制 Paddle/torch/numpy 这类底层库吃满 CPU；`0` 表示不限制。

## 主要目录

- `sensor_vector_db/config`：配置管理。
- `sensor_vector_db/core`：解析、向量化、检索、问答、参数抽取。
- `sensor_vector_db/models`：SQLite 元数据模型。
- `ui`：Streamlit Web 应用。
- `tests`：单元测试和轻量集成测试。
