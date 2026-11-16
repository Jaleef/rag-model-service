"""Cross-encoder reranker wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from app.models.runtime import resolve_model_path

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> float:
    """Convert numpy scalars/one-element arrays without deprecation warnings."""

    if isinstance(value, (int, float)):
        return float(value)

    item = getattr(value, "item", None)
    if callable(item):
        return float(item())

    return float(value)


def build_ranked_results(
    documents: Sequence[str],
    scores: Sequence[float],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Pair documents with scores and return the ``top_n`` highest scores.

    Sorting is stable, so equal scores keep the original document order.  That
    makes responses deterministic across runs and Python versions.
    """

    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    if len(documents) != len(scores):
        raise ValueError(
            "The number of scores must match the number of documents: "
            f"{len(scores)} != {len(documents)}",
        )

    results = [
        {
            "index": index,
            "text": document,
            "score": _as_float(score),
        }
        for index, (document, score) in enumerate(zip(documents, scores))
    ]

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_n]


class RerankerModel:
    """Thin, testable wrapper around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 4,
    ) -> None:
        # Imported lazily; see EmbeddingModel for the rationale.
        from sentence_transformers import CrossEncoder

        resolved_path = resolve_model_path(
            Path(model_path),
            kind="Reranker",
            env_var="RERANKER_MODEL_PATH",
        )

        logger.info("Loading reranker model from %s (device=%s)", resolved_path, device)
        self.model = CrossEncoder(
            resolved_path,
            device=device,
            max_length=max_length,
        )
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")

        if not documents:
            return []

        pairs = [(query, document) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        return build_ranked_results(documents, scores, top_n)
