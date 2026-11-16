"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.config import Settings, get_settings
from app.models.embedding import EmbeddingModel
from app.models.reranker import RerankerModel


def get_app_settings(request: Request) -> Settings:
    """Return the settings associated with the current application."""

    settings = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def get_embedding_model(request: Request) -> EmbeddingModel:
    model = getattr(request.app.state, "embedding", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not loaded.",
        )
    return model


def get_reranker_model(request: Request) -> RerankerModel:
    model = getattr(request.app.state, "reranker", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model is not loaded.",
        )
    return model
