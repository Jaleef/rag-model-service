"""Embedding model wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

from app.models.runtime import resolve_model_path

logger = logging.getLogger(__name__)

VALID_INPUT_TYPES = frozenset({"query", "document"})


class EmbeddingModel:
    """Thin, testable wrapper around a sentence-transformers model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        # Imported lazily so that importing the application (and running API
        # tests with fake models) does not require torch/transformers.
        from sentence_transformers import SentenceTransformer

        resolved_path = resolve_model_path(
            Path(model_path),
            kind="Embedding",
            env_var="EMBEDDING_MODEL_PATH",
        )

        logger.info("Loading embedding model from %s (device=%s)", resolved_path, device)
        self.model = SentenceTransformer(resolved_path, device=device)
        self.batch_size = batch_size

    def encode(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[list[float]]:
        """Encode texts and return plain Python lists.

        ``input_type`` controls whether the model's query prompt or document
        prompt is applied.  Rejecting unknown values here prevents silently
        encoding a query as a document when this class is used directly.
        """

        if input_type not in VALID_INPUT_TYPES:
            raise ValueError(
                f"input_type must be one of {sorted(VALID_INPUT_TYPES)}, "
                f"got {input_type!r}",
            )

        if not texts:
            return []

        encoder = (
            self.model.encode_query
            if input_type == "query"
            else self.model.encode_document
        )

        embeddings = encoder(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
