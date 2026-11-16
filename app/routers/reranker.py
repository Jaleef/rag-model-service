"""Rerank endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings
from app.dependencies import get_app_settings, get_reranker_model
from app.models.reranker import RerankerModel
from app.schemas.request import RerankRequest
from app.schemas.response import RerankResponse, RerankResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["reranking"])


@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank documents against a query",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not loaded"},
    },
)
def rerank(
    request: RerankRequest,
    model: RerankerModel = Depends(get_reranker_model),
    settings: Settings = Depends(get_app_settings),
) -> RerankResponse:
    try:
        results = model.rerank(
            request.query,
            request.documents,
            request.top_n,
        )
    except Exception as exc:
        logger.exception("Rerank inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rerank inference failed.",
        ) from exc

    return RerankResponse(
        model=settings.reranker_model_id,
        results=[RerankResult(**result) for result in results],
    )
