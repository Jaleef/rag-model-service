"""API tests using fake models (no torch/sentence-transformers required)."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeEmbeddingModel:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

    def encode(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        if "boom" in texts:
            raise RuntimeError("embedding failure")

        base = 1.0 if input_type == "query" else 0.5
        return [[base, float(len(text))] for text in texts]


class FakeRerankerModel:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if "boom" in documents:
            raise RuntimeError("rerank failure")

        results = [
            {
                "index": index,
                "text": document,
                "score": float(len(document)),
            }
            for index, document in enumerate(documents)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_n]


def make_settings() -> Settings:
    return Settings(
        model_dir=Path("models"),
        embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
        reranker_model_id="Qwen/Qwen3-Reranker-0.6B",
        device="cpu",
    )


def build_app():
    return create_app(
        make_settings(),
        embedding_factory=FakeEmbeddingModel,
        reranker_factory=FakeRerankerModel,
        runtime_configurator=lambda settings: None,
    )


class ApiTests(unittest.TestCase):
    def test_health(self) -> None:
        with TestClient(build_app()) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_reports_loaded_models(self) -> None:
        with TestClient(build_app()) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ready",
                "models": {
                    "embedding": "Qwen/Qwen3-Embedding-0.6B",
                    "reranker": "Qwen/Qwen3-Reranker-0.6B",
                },
            },
        )

    def test_embeddings_success(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": ["a", "bb"], "input_type": "query"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(
            [item["index"] for item in payload["data"]],
            [0, 1],
        )
        self.assertEqual(payload["data"][0]["embedding"], [1.0, 1.0])
        self.assertEqual(payload["data"][1]["embedding"], [1.0, 2.0])

    def test_embeddings_reject_invalid_input_type(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": ["a"], "input_type": "passage"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("input_type", response.json()["detail"])

    def test_embeddings_reject_empty_and_oversized_batches(self) -> None:
        with TestClient(build_app()) as client:
            empty = client.post("/v1/embeddings", json={"input": []})
            oversized = client.post(
                "/v1/embeddings",
                json={"input": ["x"] * 129},
            )

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(oversized.status_code, 422)

    def test_embeddings_inference_error_returns_500(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": ["boom"]},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Embedding inference failed.")

    def test_rerank_sorts_and_applies_top_n(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/rerank",
                json={
                    "query": "query",
                    "documents": ["a", "longer", "mid"],
                    "top_n": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "Qwen/Qwen3-Reranker-0.6B")
        self.assertEqual(
            [item["index"] for item in payload["results"]],
            [1, 2],
        )
        self.assertEqual(
            [item["score"] for item in payload["results"]],
            [6.0, 3.0],
        )

    def test_rerank_top_n_larger_than_documents(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/rerank",
                json={
                    "query": "query",
                    "documents": ["a", "bb"],
                    "top_n": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 2)

    def test_rerank_validation(self) -> None:
        with TestClient(build_app()) as client:
            empty_query = client.post(
                "/v1/rerank",
                json={"query": "", "documents": ["a"]},
            )
            empty_documents = client.post(
                "/v1/rerank",
                json={"query": "q", "documents": []},
            )
            invalid_top_n = client.post(
                "/v1/rerank",
                json={"query": "q", "documents": ["a"], "top_n": 0},
            )

        self.assertEqual(empty_query.status_code, 422)
        self.assertEqual(empty_documents.status_code, 422)
        self.assertEqual(invalid_top_n.status_code, 422)

    def test_rerank_inference_error_returns_500(self) -> None:
        with TestClient(build_app()) as client:
            response = client.post(
                "/v1/rerank",
                json={"query": "q", "documents": ["boom"]},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Rerank inference failed.")

    def test_endpoints_return_503_when_models_are_missing(self) -> None:
        app = create_app(
            make_settings(),
            embedding_factory=lambda *args, **kwargs: None,
            reranker_factory=lambda *args, **kwargs: None,
            runtime_configurator=lambda settings: None,
        )

        with TestClient(app) as client:
            ready = client.get("/ready")
            embeddings = client.post("/v1/embeddings", json={"input": ["a"]})
            rerank = client.post(
                "/v1/rerank",
                json={"query": "q", "documents": ["d"]},
            )

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(embeddings.status_code, 503)
        self.assertEqual(rerank.status_code, 503)


if __name__ == "__main__":
    unittest.main()
