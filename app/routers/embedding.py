"""Embedding endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings
from app.dependencies import get_app_settings, get_embedding_model
from app.models.embedding import VALID_INPUT_TYPES, EmbeddingModel
from app.schemas.request import EmbeddingRequest
from app.schemas.response import EmbeddingData, EmbeddingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings for a batch of texts",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid input_type"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not loaded"},
    },
)
def create_embeddings(
    request: EmbeddingRequest,
    model: EmbeddingModel = Depends(get_embedding_model),
    settings: Settings = Depends(get_app_settings),
) -> EmbeddingResponse:
    # The schema keeps ``input_type`` as a string so that an invalid value is
    # reported as 400 (the documented contract) instead of FastAPI's generic
    # 422 validation error.
    if request.input_type not in VALID_INPUT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input_type must be 'query' or 'document'.",
        )

    try:
        vectors = model.encode(request.input, request.input_type)
    except Exception as exc:
        logger.exception("Embedding inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding inference failed.",
        ) from exc

    return EmbeddingResponse(
        model=settings.embedding_model_id,
        data=[
            EmbeddingData(index=index, embedding=vector)
            for index, vector in enumerate(vectors)
        ],
    )
