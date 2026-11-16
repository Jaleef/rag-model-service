"""Tests for settings and model path handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.models.runtime import resolve_model_path


class SettingsTests(unittest.TestCase):
    def test_model_paths_are_derived_from_ids(self) -> None:
        settings = Settings(
            model_dir=Path("models"),
            embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
            reranker_model_id="Qwen/Qwen3-Reranker-0.6B",
        )

        self.assertEqual(
            settings.embedding_path,
            Path("models") / "Qwen3-Embedding-0.6B",
        )
        self.assertEqual(
            settings.reranker_path,
            Path("models") / "Qwen3-Reranker-0.6B",
        )

    def test_explicit_model_paths_take_precedence(self) -> None:
        settings = Settings(
            model_dir=Path("models"),
            embedding_model_path=Path("custom/embedding"),
            reranker_model_path=Path("custom/reranker"),
        )

        self.assertEqual(settings.embedding_path, Path("custom/embedding"))
        self.assertEqual(settings.reranker_path, Path("custom/reranker"))

    def test_empty_model_path_is_none(self) -> None:
        settings = Settings(embedding_model_path="")

        self.assertIsNone(settings.embedding_model_path)

    def test_model_dir_expands_user(self) -> None:
        settings = Settings(model_dir="~/rag-models")

        self.assertEqual(settings.model_dir, Path("~/rag-models").expanduser())


class ResolveModelPathTests(unittest.TestCase):
    def test_existing_directory_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()

            resolved = resolve_model_path(
                model_dir,
                kind="Embedding",
                env_var="EMBEDDING_MODEL_PATH",
            )

            self.assertEqual(resolved, str(model_dir))

    def test_missing_rooted_path_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            with self.assertRaises(FileNotFoundError) as context:
                resolve_model_path(
                    missing,
                    kind="Embedding",
                    env_var="EMBEDDING_MODEL_PATH",
                )

            self.assertIn("download_models", str(context.exception))

    def test_relative_value_can_be_a_hugging_face_id(self) -> None:
        resolved = resolve_model_path(
            Path("Qwen/Qwen3-Embedding-0.6B"),
            kind="Embedding",
            env_var="EMBEDDING_MODEL_PATH",
        )

        self.assertEqual(resolved, "Qwen/Qwen3-Embedding-0.6B")


if __name__ == "__main__":
    unittest.main()
