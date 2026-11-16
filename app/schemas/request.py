"""Request schemas for the public HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_EMBEDDING_INPUTS = 128
MAX_RERANK_DOCUMENTS = 50


class EmbeddingRequest(BaseModel):
    """Payload for ``POST /v1/embeddings``."""

    input: list[str] = Field(
        min_length=1,
        max_length=MAX_EMBEDDING_INPUTS,
        description=f"Texts to embed (1-{MAX_EMBEDDING_INPUTS}).",
    )
    input_type: str = Field(
        default="document",
        description=(
            "Use 'query' for retrieval queries and 'document' for passages. "
            "Invalid values are rejected with HTTP 400."
        ),
        examples=["query", "document"],
    )


class RerankRequest(BaseModel):
    """Payload for ``POST /v1/rerank``."""

    query: str = Field(
        min_length=1,
        description="The search query.",
    )
    documents: list[str] = Field(
        min_length=1,
        max_length=MAX_RERANK_DOCUMENTS,
        description=f"Candidate documents (1-{MAX_RERANK_DOCUMENTS}).",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=MAX_RERANK_DOCUMENTS,
        description="Number of highest-scoring documents to return.",
    )
