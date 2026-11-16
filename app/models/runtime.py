"""Process-wide runtime helpers for the model layer."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

_torch_configured = False
_torch_lock = threading.Lock()


def configure_torch(settings: Settings) -> None:
    """Configure PyTorch thread pools exactly once per process.

    ``torch.set_num_interop_threads`` may only be called before parallel work
    starts and raises ``RuntimeError`` when called twice.  The previous
    implementation called it at import time, which made the module fragile
    under test runners, reloaders and forked workers.
    """

    global _torch_configured

    with _torch_lock:
        if _torch_configured:
            return

        import torch

        torch.set_num_threads(settings.torch_num_threads)

        try:
            torch.set_num_interop_threads(settings.torch_num_interop_threads)
        except RuntimeError:
            # Parallel interop work has already started (e.g. another
            # framework initialized the pool first).  Keep the current value
            # instead of crashing the whole service at startup.
            logger.warning(
                "PyTorch interop threads were already initialized; "
                "keeping the existing configuration.",
            )

        _torch_configured = True
        logger.info(
            "PyTorch threads configured: intra=%d, interop=%d",
            settings.torch_num_threads,
            settings.torch_num_interop_threads,
        )


def resolve_model_path(path: Path, *, kind: str, env_var: str) -> str:
    """Return a path or Hugging Face id accepted by sentence-transformers.

    Absolute (or otherwise rooted) paths must exist so that a missing Docker
    volume produces an actionable error instead of a confusing download
    attempt.  Relative values are passed through and may be repository ids
    such as ``Qwen/Qwen3-Embedding-0.6B``.
    """

    if path.exists():
        return str(path)

    if path.anchor:
        raise FileNotFoundError(
            f"{kind} model directory not found: {path}. "
            "Download the models with "
            "`uv run python -m scripts.download_models` or set "
            f"{env_var} to a valid model directory."
        )

    logger.warning(
        "%s model path %s does not exist; treating it as a Hugging Face model id.",
        kind,
        path,
    )
    # Keep forward slashes so repository ids work on Windows too.
    return path.as_posix()
