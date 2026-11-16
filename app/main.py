"""FastAPI application entry point.

The application is assembled by :func:`create_app` so that tests can inject
fake models and skip the heavy PyTorch runtime configuration.  Importing this
module does not import torch or sentence-transformers; those imports happen
when the real model wrappers are constructed during the lifespan startup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from app.config import Settings, get_settings
from app.models.embedding import EmbeddingModel
from app.models.reranker import RerankerModel
from app.models.runtime import configure_torch
from app.routers import embedding_router, reranker_router
from app.schemas.request import EmbeddingRequest, RerankRequest
from app.schemas.response import HealthResponse, ReadyResponse

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility with code that imported the request
# models from app.main.
__all__ = [
    "EmbeddingRequest",
    "RerankRequest",
    "app",
    "create_app",
    "main",
]


def _release_models(application: FastAPI) -> None:
    application.state.embedding = None
    application.state.reranker = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings: Settings = application.state.settings
    configurator: Callable[[Settings], None] = application.state.runtime_configurator

    configurator(settings)

    try:
        logger.info("Loading embedding model from %s", settings.embedding_path)
        embedding = application.state.embedding_factory(
            settings.embedding_path,
            device=settings.device,
            batch_size=settings.embedding_batch_size,
        )
        application.state.embedding = embedding

        logger.info("Loading reranker model from %s", settings.reranker_path)
        reranker = application.state.reranker_factory(
            settings.reranker_path,
            device=settings.device,
            batch_size=settings.rerank_batch_size,
            max_length=settings.rerank_max_length,
        )
        application.state.reranker = reranker
    except Exception:
        # Do not leave a half-initialized model set behind when the second
        # model fails to load.
        _release_models(application)
        logger.exception("Failed to initialize models")
        raise

    logger.info("Models loaded successfully")
    try:
        yield
    finally:
        _release_models(application)
        # Drop the local references so the models can be garbage collected.
        embedding = None  # noqa: F841 - explicitly release references
        reranker = None  # noqa: F841
        logger.info("Models released")


def create_app(
    settings: Settings | None = None,
    *,
    embedding_factory: Callable[..., Any] = EmbeddingModel,
    reranker_factory: Callable[..., Any] = RerankerModel,
    runtime_configurator: Callable[[Settings], None] = configure_torch,
) -> FastAPI:
    """Build and return a configured FastAPI application."""

    settings = settings or get_settings()

    application = FastAPI(
        title="RAG Model Service",
        description=(
            "CPU-friendly embedding and reranking service for "
            "Qwen3-Embedding and Qwen3-Reranker."
        ),
        version="1.1.0",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.embedding_factory = embedding_factory
    application.state.reranker_factory = reranker_factory
    application.state.runtime_configurator = runtime_configurator
    application.state.embedding = None
    application.state.reranker = None

    application.include_router(embedding_router)
    application.include_router(reranker_router)

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Liveness probe",
    )
    def health() -> HealthResponse:
        return HealthResponse()

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["system"],
        summary="Readiness probe (both models loaded)",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Models not loaded"},
        },
    )
    def ready(request: Request) -> ReadyResponse:
        embedding = getattr(request.app.state, "embedding", None)
        reranker = getattr(request.app.state, "reranker", None)
        if embedding is None or reranker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models are not loaded.",
            )

        return ReadyResponse(
            models={
                "embedding": settings.embedding_model_id,
                "reranker": settings.reranker_model_id,
            },
        )

    return application


app = create_app()


def main() -> None:
    """Console-script / ``python -m app.main`` entry point."""

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
