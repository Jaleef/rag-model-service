"""Application configuration.

All settings can be overridden with environment variables (case-insensitive)
or with a local ``.env`` file.  For example::

    MODEL_DIR=/opt/models
    DEVICE=cpu
    TORCH_NUM_THREADS=4
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the RAG model service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # --- Model locations -------------------------------------------------
    # ``*_model_id`` is a Hugging Face repository id (or a model name).
    # The model is loaded from ``MODEL_DIR / <repository name>`` unless an
    # explicit ``*_MODEL_PATH`` is provided.
    model_dir: Path = Path("/models")
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    embedding_model_path: Path | None = None
    reranker_model_path: Path | None = None
    device: str = "cpu"

    # --- Runtime tuning ---------------------------------------------------
    torch_num_threads: int = Field(default=4, ge=1)
    torch_num_interop_threads: int = Field(default=1, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    rerank_batch_size: int = Field(default=4, ge=1)
    rerank_max_length: int = Field(default=512, ge=1)

    # --- HTTP server ------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    @field_validator(
        "embedding_model_path",
        "reranker_model_path",
        mode="before",
    )
    @classmethod
    def _empty_path_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_dir", mode="after")
    @classmethod
    def _expand_model_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator(
        "embedding_model_path",
        "reranker_model_path",
        mode="after",
    )
    @classmethod
    def _expand_model_path(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @property
    def embedding_path(self) -> Path:
        """Local directory of the embedding model."""
        if self.embedding_model_path is not None:
            return self.embedding_model_path
        return self.model_dir / Path(self.embedding_model_id).name

    @property
    def reranker_path(self) -> Path:
        """Local directory of the reranker model."""
        if self.reranker_model_path is not None:
            return self.reranker_model_path
        return self.model_dir / Path(self.reranker_model_id).name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached).

    Tests or callers that change environment variables at runtime can call
    ``get_settings.cache_clear()`` before requesting new settings.
    """
    return Settings()
