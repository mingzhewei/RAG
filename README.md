# 本地传感器文档智能助手

基于本地向量数据库的传感器技术文档 RAG（检索增强生成）系统。支持 PDF、Word、文本文档和代码文件的导入，通过混合检索（语义 + 关键词）找到相关内容，由大模型基于文档证据生成回答。

## 快速启动

```powershell
# 1. 安装依赖
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. 配置环境
Copy-Item .env.example .env
# 编辑 .env，填入你的大模型 API Key

# 3. 启动
.\.venv\Scripts\python main.py
```

只检查环境、不启动界面：

```powershell
.\.venv\Scripts\python main.py --check
```

指定端口：

```powershell
.\.venv\Scripts\python main.py --port 8502
```

## 页面功能

### 首页 — 智能搜索

统一搜索入口，输入问题即可获取 AI 回答。支持：

- 自然语言提问，AI 基于已导入的文档回答
- 对话历史，可连续追问
- 参考来源展示，每个回答附带文档出处
- 关键词高亮
- 快捷示例按钮

侧边栏包含系统状态（文档数、片段数、向量数）和功能导航。高级设置（检索片段数、检索策略）默认折叠。

### 文档导入

将传感器技术文档导入本地向量数据库：

- 支持 PDF、Word (.docx)、文本 (.txt/.md) 和代码文件
- 自动识别新增、修改和已删除文件
- 哈希去重：相同内容的文件只索引一次
- 断点续传：中断后可恢复，已完成的文件自动跳过
- 实时进度展示和文件明细

### 智能检索

直接搜索文档内容，查看原始匹配片段。支持：

- 三种检索模式：混合检索（推荐）、语义检索、关键词检索
- 按文件类型、传感器型号、制造商筛选
- 结果按相关度排序，含来源文件和页码

### 智能问答

多轮对话式问答，使用 Streamlit 原生聊天组件。AI 会基于已导入的文档回答，每条结论标注来源编号。

### 参数抽取

从已入库文档中自动抽取传感器技术参数（量程、精度、分辨率、工作温度等），支持 LLM 校验字段名。

### 参数对比

选择 2-5 个传感器型号，生成跨型号参数对比表格，可导出 Markdown 或 CSV。

### 系统管理

- 查看系统状态和存储位置
- 切换大模型配置（CRS / DeepSeek / 禁用）
- 刷新文档元数据（型号、厂商识别）
- 查看嵌入模型、OCR 等处理参数

## 大模型配置

密钥写入 `.env` 文件，或在系统管理页面临时输入。默认配置：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | `crs` | 提供方：`crs`、`deepseek`、`none` |
| `WIRE_API` | `responses` | API 协议：`responses` 或 `chat_completions` |
| `CRS` | `gpt-5.5` | CRS 模型名 |

切换到 DeepSeek 时设置 `LLM_PROVIDER=deepseek`，禁用生成时设为 `LLM_PROVIDER=none`。

## 运行测试

测试使用确定性 fake embedding，不需要下载 BGE-M3 模型或调用外部 API：

```powershell
.\.venv\Scripts\pytest
```

## 导入范围

- `.txt` 仅导入不大于 1 MiB 的文件
- CSV、PCAP、音视频、数据库文件一律排除

## 持久导入

- 文档导入页使用持久任务表记录目录同步进度
- 同一目录再次同步时按文件哈希识别新增、修改和未变化文件
- 源目录删除的文件会在下次同步时清理对应元数据和向量
- 关机或服务重启导致任务中断后，回到文档导入页选择任务并点击"恢复任务"即可继续
- 单文件 checkpoint 机制保证向量化中断后只补齐缺失向量，不重新解析
- 点击"停止任务"或 `Ctrl+C` 时，停止请求先写入 SQLite，当前批次完成后中断

## 导入加速建议

- 默认关闭 OCR：`OCR_ENABLED=false`。先导入可复制文本的 PDF
- 文字版入库后，设置 `OCR_ENABLED=true` 再恢复导入，自动重建需要 OCR 的文件
- `EMBEDDING_DEVICE=cuda` + `EMBEDDING_USE_FP16=true` 可利用 GPU 加速向量化
- `CHUNK_SIZE` 调大减少向量数量但降低检索粒度

## 项目结构

```
sensor_vector_db/
  config/      配置管理（Settings）
  core/        文档解析、向量化、检索、问答、参数抽取
  models/      SQLite 数据模型
  utils/       日志、文件工具、哈希工具
ui/
  app.py       首页（智能搜索入口）
  bootstrap.py 服务缓存和配置桥接
  components/  渲染组件
  pages/       功能页面（文档导入、检索、问答、参数抽取/对比、系统管理）
tests/         单元测试和集成测试
main.py        启动入口
```

## 准确性原则

- 回答必须严格基于检索到的文档片段
- 参数值保留来源文本、页码信息
- 未找到依据时显示"未在已入库文档中找到依据"，不推断、不补全
