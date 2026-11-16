"""Download the models required by the service.

Usage::

    uv run python -m scripts.download_models
    uv run python -m scripts.download_models --model embedding
    uv run python -m scripts.download_models --revision main

The default target directory is ``MODEL_DIR`` (``./models``).  In the
container the same models live under ``/models``.  Set ``HF_TOKEN`` for
gated/private repositories.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow ``python scripts/download_models.py`` without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings  # noqa: E402

logger = logging.getLogger("download_models")

MODEL_CHOICES = ("all", "embedding", "reranker")


def download_model(
    repo_id: str,
    target_dir: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a Hugging Face repository into ``target_dir``."""

    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", repo_id, target_dir)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(target_dir),
    )

    return target_dir


def _download_selected(
    settings: Settings,
    model: str,
    *,
    revision: str | None,
    token: str | None,
) -> None:
    if model in ("all", "embedding"):
        download_model(
            settings.embedding_model_id,
            settings.embedding_path,
            revision=revision,
            token=token,
        )

    if model in ("all", "reranker"):
        download_model(
            settings.reranker_model_id,
            settings.reranker_path,
            revision=revision,
            token=token,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Qwen3 embedding/reranker models.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="all",
        help="Which model to download (default: all).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision (branch, tag or commit).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; defaults to HF_TOKEN.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    token = args.token or os.environ.get("HF_TOKEN") or None
    settings = get_settings()

    try:
        _download_selected(
            settings,
            args.model,
            revision=args.revision,
            token=token,
        )
    except Exception:
        logger.exception("Model download failed")
        return 1

    logger.info("Done. Models are available under %s", settings.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
