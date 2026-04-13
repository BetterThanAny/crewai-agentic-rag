# 🤖 CrewAI Agentic RAG

基于 CrewAI 的多 Agent 协作检索增强生成（RAG）系统。上传文档后，三个 AI Agent 协同完成意图路由、知识检索和回答生成。

## 架构

```
用户问题
   │
   ▼
┌─────────────┐     RETRIEVE     ┌─────────────┐     ┌─────────────┐
│   Router    │ ───────────────▶ │  Retriever  │ ──▶ │  Responder  │ ──▶ 回答
│  意图路由器  │                  │  知识检索员   │     │  回答生成器  │
└─────────────┘                  └──────┬──────┘     └─────────────┘
       │                                │
       │ DIRECT                   向量检索 (ChromaDB)
       │                                │
       └──────────────────────────▶ Responder ──▶ 直接回答
```

**执行流程：**

1. **Router** — 判断问题是否需要检索文档（`RETRIEVE`）还是可以直接回答（`DIRECT`）
2. **Retriever** — 使用向量检索工具从 ChromaDB 中查找相关文档片段
3. **Responder** — 整合检索结果，生成带引用来源的中文回答

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Agent 框架 | CrewAI >= 1.14 | 多 Agent 编排、memory、顺序执行 |
| LLM | DeepSeek / Qwen | OpenAI 兼容格式，通过 `base_url` 切换 |
| Embedding | 阿里 text-embedding-v3 | 可回退到 ChromaDB 内置模型 |
| 向量库 | ChromaDB | 零配置持久化，cosine 相似度 |
| 文档处理 | PyMuPDF | PDF / TXT / Markdown 多格式支持 |
| 前端 | Streamlit | 聊天界面 + 文档上传 + Agent 可视化 |
| 包管理 | uv | 快速依赖管理 |
| 容器化 | Docker | 一键部署 |

## 快速开始

### 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone https://github.com/your-username/crewai-agentic-rag.git
cd crewai-agentic-rag

# 安装依赖
uv sync
```

### 配置

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=your-key-here

# 可选：Embedding 配置（不配置则使用本地模型）
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=your-key-here
```

### 灌入文档

将文档放入 `data/` 目录，然后执行：

```bash
uv run python ingest.py --reset
```

### 使用方式

**Web 界面（推荐）：**

```bash
uv run streamlit run app.py
```

访问 `http://localhost:8501`，在侧边栏上传文档，然后开始提问。

**命令行 — 单次查询：**

```bash
uv run python main.py --query "什么是 Python 装饰器？"
```

**命令行 — 交互模式：**

```bash
uv run python main.py
```

### Docker 部署

```bash
docker compose up --build
```

访问 `http://localhost:8501`。

## 项目结构

```
crewai-agentic-rag/
├── src/
│   ├── config.py              # LLM & Embedding 统一配置
│   ├── document_loader.py     # 多格式文档加载（PDF/TXT/MD）
│   ├── chunker.py             # 递归字符文本切片
│   ├── vector_store.py        # ChromaDB 向量库封装
│   ├── agents.py              # Router/Retriever/Responder Agent 定义
│   ├── tasks.py               # Task 定义与上下文传递
│   ├── crew.py                # Crew 组装与执行流水线
│   └── tools/
│       └── vector_search_tool.py  # 向量检索 CrewAI Tool
├── tests/                     # 测试套件（72 个测试）
├── data/                      # 示例文档
├── app.py                     # Streamlit Web UI
├── main.py                    # CLI 入口
├── ingest.py                  # 文档灌入脚本
├── Dockerfile
└── docker-compose.yml
```

## 测试

```bash
# 快速测试（mock，不消耗 API）
uv run pytest tests/ -v -m "not slow"

# 完整测试（需要真实 API key）
uv run pytest tests/ -v
```

## 关键设计决策

1. **两阶段路由** — Router 先判断问题类型再决定是否检索，避免对常识问题做不必要的向量查询，节省 token 和时间。

2. **顺序执行（Process.sequential）** — 三个 Agent 严格按 Router → Retriever → Responder 顺序执行，通过 Task 的 `context` 参数传递上游结果，保证信息流的正确性。

3. **CrewAI Memory** — 启用短期记忆支持多轮对话，后续追问能引用之前的上下文。

4. **向量检索优雅降级** — `vector_search_tool` 在向量库不可用时自动回退到 mock 数据，保证 Agent 层可独立开发和测试。

5. **LLM 多 Provider 支持** — 通过 `.env` 配置切换 DeepSeek / Qwen / OpenAI 兼容中转站 / Ollama，不需要改代码。

6. **递归字符切片** — `chunk_size=500, overlap=100`，按 `\n\n` → `\n` → `。` → `.` → 空格逐级细分，兼顾中英文文档。

## 许可证

MIT
