## JimmerRAG

基于 LangChain 与 HuggingFace Embeddings 的检索增强生成（RAG）系统，用于快速查询 Jimmer 框架文档。

### 环境准备
- Python 3.13
- 安装依赖：`uv pip install -r pyproject.toml` 或使用 `pip install -r requirements.txt`（可自行生成）
- HuggingFace 模型会自动下载；首次运行可能需要较长时间。
- 需要 OpenAI 兼容接口，设置 `OPENAI_API_KEY`（如使用官方 OpenAI，需 `pip install openai` 已在依赖内）。
- 如使用阿里灵积 DashScope 等 OpenAI 兼容服务，可设置 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`，并指定 `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`（或 `DASHSCOPE_BASE_URL`），模型名称通过 `CHAT_MODEL`/`DASHSCOPE_MODEL` 环境变量配置（例如 `qwen-plus`）。
- 如需重新解析 MDX，可运行 `npm install` 于 `src/node` 目录（可选）。

### 构建与查询
```bash
python -m src.main --skip-clone --rebuild-index
python -m src.main --skip-clone --question "Jimmer 是什么？"
```

常用参数：
- `--question`：执行问答；若省略则只构建/加载向量库。
- `--rebuild-index`：强制重新生成向量索引。
- `--skip-clone`：跳过文档仓库克隆（本地已有数据时使用）。
- `--top-k`：自定义检索文档数量。

### API 用法
代码中提供了 `RAGService` 类，可通过以下方式集成：
```python
from src.rag import RAGService

service = RAGService()
result = service.ask("Jimmer 是什么？")
print(result.answer)
print(result.sources)
```

### 调试
- `python -m src.main --help` 查看所有 CLI 选项。
- `python src/call.py "你的问题"` 运行示例脚本。


