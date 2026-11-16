
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.models.embedding import EmbeddingModel
from app.models.reranker import RerankerModel


# CPU 线程控制，避免两个模型过度争抢 CPU
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Embedding model...")
    app.state.embedding = EmbeddingModel(
        "/models/Qwen3-Embedding-0.6B"
    )

    print("Loading Reranker model...")
    app.state.reranker = RerankerModel(
        "/models/Qwen3-Reranker-0.6B"
    )

    print("Models loaded successfully!")

    yield

    # 应用关闭时释放模型引用
    app.state.embedding = None
    app.state.reranker = None


app = FastAPI(
    title="RAG Model Service",
    version="1.0.0",
    lifespan=lifespan,
)


def main():
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


class EmbeddingRequest(BaseModel):
    input: list[str] = Field(min_length=1, max_length=128)
    input_type: str = "document"


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1, max_length=50)
    top_n: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    if request.input_type not in ("query", "document"):
        raise HTTPException(
            status_code=400,
            detail="input_type must be query or document",
        )

    vectors = app.state.embedding.encode(
        request.input,
        request.input_type,
    )

    return {
        "model": "Qwen3-Embedding-0.6B",
        "data": [
            {
                "index": i,
                "embedding": vector,
            }
            for i, vector in enumerate(vectors)
        ],
    }


@app.post("/v1/rerank")
def rerank(request: RerankRequest):
    results = app.state.reranker.rerank(
        request.query,
        request.documents,
        request.top_n,
    )

    return {
        "model": "Qwen3-Reranker-0.6B",
        "results": results,
    }
