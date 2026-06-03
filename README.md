# 本地传感器 RAG 检索 MVP

这是一个本地运行的传感器技术文档 RAG 检索系统。系统以本地向量库为核心，支持 PDF、DOCX、TXT 和代码文件导入，扫描 PDF 可启用 PaddleOCR，本地 embedding 默认使用 `BAAI/bge-m3`，问答生成通过 DeepSeek Chat API 完成。

## 运行

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python main.py
.\.venv\Scripts\streamlit run ui\app.py
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

## 主要目录

- `sensor_vector_db/config`：配置管理。
- `sensor_vector_db/core`：解析、向量化、检索、问答、参数抽取。
- `sensor_vector_db/models`：SQLite 元数据模型。
- `ui`：Streamlit Web 应用。
- `tests`：单元测试和轻量集成测试。

