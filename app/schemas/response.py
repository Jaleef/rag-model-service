"""Response schemas for the public HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    model: str
    data: list[EmbeddingData]


class RerankResult(BaseModel):
    index: int
    text: str
    score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready"] = "ready"
    models: dict[str, str] = Field(default_factory=dict)
