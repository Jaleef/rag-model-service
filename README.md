# RAG Model Service

基于 FastAPI + sentence-transformers 的轻量 RAG 模型服务，面向 CPU 部署，默认加载：

- Embedding：`Qwen/Qwen3-Embedding-0.6B`
- Reranker：`Qwen/Qwen3-Reranker-0.6B`

提供以下接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/embeddings` | 批量文本向量化，支持 query / document 两种输入类型 |
| `POST` | `/v1/rerank` | 对候选文档按 query 相关性重排序 |
| `GET` | `/health` | 存活探针 |
| `GET` | `/ready` | 就绪探针（两个模型均已加载） |
| `GET` | `/docs` | Swagger UI |

## 目录结构

```text
app/
├── config.py            # pydantic-settings 配置（模型路径、线程、批大小等）
├── dependencies.py      # 模型 / 配置依赖，模型未加载时返回 503
├── main.py              # create_app、lifespan、健康检查、启动入口
├── models/
│   ├── embedding.py     # EmbeddingModel（延迟导入 sentence-transformers）
│   ├── reranker.py      # RerankerModel + 纯函数排序逻辑
│   └── runtime.py       # PyTorch 线程配置、模型路径解析
├── routers/
│   ├── embedding.py     # POST /v1/embeddings
│   └── reranker.py      # POST /v1/rerank
└── schemas/
    ├── request.py       # 请求体校验
    └── response.py      # 响应体
scripts/
└── download_models.py   # 从 Hugging Face 下载模型
tests/                   # 使用 fake model 的单元/接口测试
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 下载模型

```bash
# 两个模型都下载到 MODEL_DIR（默认 /models）
uv run python -m scripts.download_models

# 只下载其中一个
uv run python -m scripts.download_models --model embedding
uv run python -m scripts.download_models --model reranker
```

在 Windows / macOS 本地开发时，建议把模型放到项目目录并设置 `MODEL_DIR`：

```powershell
$env:MODEL_DIR = "./models"
uv run python -m scripts.download_models
```

目录结构应为：

```text
models/
├── Qwen3-Embedding-0.6B/
└── Qwen3-Reranker-0.6B/
```

### 3. 启动服务

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

或使用安装后的命令行入口：

```bash
uv run rag-model-service
```

## Docker

```bash
uv run python -m scripts.download_models
docker compose up --build
```

`docker-compose.yml` 会把 `./models` 只读挂载到容器内的 `/models`。镜像内置健康检查，`/ready` 在模型未加载完成时返回 `503`。

## API 示例

### Embedding

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["什么是向量检索？", "向量检索是一种语义检索方式。"],
    "input_type": "query"
  }'
```

```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B",
  "data": [
    {"index": 0, "embedding": [0.01, -0.02, "..."]},
    {"index": 1, "embedding": [0.03, 0.04, "..."]}
  ]
}
```

- `input`：1–128 条文本。
- `input_type`：`query` 或 `document`，默认 `document`；非法值返回 `400`。
- 向量已做 L2 归一化，可直接用点积 / 余弦相似度比较。

### Rerank

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是向量检索？",
    "documents": ["向量检索是一种语义检索方式。", "今天天气不错。"],
    "top_n": 2
  }'
```

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "results": [
    {"index": 0, "text": "向量检索是一种语义检索方式。", "score": 8.42},
    {"index": 1, "text": "今天天气不错。", "score": -9.17}
  ]
}
```

- `documents`：1–50 条候选文档。
- `top_n`：1–50，默认 5；实际返回数量不会超过文档数。
- `score` 是 CrossEncoder 的原始相关性分数，数值范围由模型决定，排序时只需要看相对大小。
- 分数相同时保持输入文档顺序，保证结果稳定。

## 配置

所有配置均可通过环境变量或 `.env` 覆盖（参考 `.env.example`）：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_DIR` | `/models` | 模型根目录 |
| `EMBEDDING_MODEL_ID` | `Qwen/Qwen3-Embedding-0.6B` | embedding 仓库 id / 响应中的模型名 |
| `RERANKER_MODEL_ID` | `Qwen/Qwen3-Reranker-0.6B` | reranker 仓库 id / 响应中的模型名 |
| `EMBEDDING_MODEL_PATH` | 空 | 显式指定 embedding 目录，优先级高于 `MODEL_DIR/<repo name>` |
| `RERANKER_MODEL_PATH` | 空 | 显式指定 reranker 目录 |
| `DEVICE` | `cpu` | `cpu`、`cuda`、`cuda:0`、`mps` 等 |
| `TORCH_NUM_THREADS` | `4` | PyTorch 单次推理线程数 |
| `TORCH_NUM_INTEROP_THREADS` | `1` | PyTorch interop 线程数（进程内只设置一次） |
| `EMBEDDING_BATCH_SIZE` | `32` | embedding 批大小 |
| `RERANK_BATCH_SIZE` | `4` | rerank 批大小 |
| `RERANK_MAX_LENGTH` | `512` | reranker 最大序列长度 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 测试

```bash
uv run pytest
```

测试使用 fake model，不会加载真实权重，因此不需要下载模型；覆盖了：

- 配置派生 / 路径解析 / 空值处理
- 排序、`top_n`、同分稳定性、异常输入
- 接口成功、400/422/500/503 分支、健康检查

## 本次重构与修复要点

1. **配置化**：删除 `/models/...` 硬编码，新增 `Settings`，模型路径、设备、线程、批大小均可配置。
2. **延迟导入重型依赖**：`torch`、`sentence-transformers` 只在真正构造模型时导入，应用导入和接口测试不再依赖数 GB 的运行时。
3. **PyTorch 线程配置幂等**：`torch.set_num_interop_threads()` 从模块导入期移到 lifespan，并处理“已初始化”导致的 `RuntimeError`。
4. **修复 Docker 构建失败**：`uv sync` 在 `COPY app` 之前执行会因为构建后端找不到 `app/` 而失败；改为 `--no-install-project`，只安装第三方依赖。
5. **资源清理**：任一模型加载失败时释放已加载模型；关闭时清空引用，避免半初始化状态。
6. **缺失模型的可诊断错误**：模型目录不存在时给出包含下载命令的错误信息，而不是让 sentence-transformers 去联网猜仓库。
7. **接口拆分与校验**：请求/响应 schema、路由、依赖拆到独立模块；补充批次上限、`top_n`、空 query 校验，模型未加载时返回 `503`。
8. **确定性排序**：分数相同时按原始文档顺序返回。
9. **部署细节**：新增 `.dockerignore`（避免把 `models/` 打进构建上下文）、镜像健康检查、`.env.example` 和模型下载脚本。
10. **测试**：新增配置、排序和 API 测试，覆盖主要失败路径。

> 说明：Qwen3-Reranker 通过 sentence-transformers 的 `CrossEncoder`（`LogitScore`）加载，需要较新的 sentence-transformers；本项目锁定版本为 `6.3.0`。
