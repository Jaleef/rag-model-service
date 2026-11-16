"""Tests for the pure reranking helpers (no torch required)."""

from __future__ import annotations

import unittest

from app.models.reranker import build_ranked_results


class BuildRankedResultsTests(unittest.TestCase):
    def test_sorts_by_score_descending(self) -> None:
        results = build_ranked_results(
            ["a", "bb", "ccc"],
            [3.0, 1.0, 2.0],
        )

        self.assertEqual([item["index"] for item in results], [0, 2, 1])
        self.assertEqual([item["score"] for item in results], [3.0, 2.0, 1.0])

    def test_applies_top_n(self) -> None:
        results = build_ranked_results(
            ["a", "bb", "ccc"],
            [3.0, 1.0, 2.0],
            top_n=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([item["index"] for item in results], [0, 2])

    def test_ties_keep_original_document_order(self) -> None:
        results = build_ranked_results(
            ["first", "second", "third"],
            [1.0, 1.0, 1.0],
        )

        self.assertEqual(
            [item["index"] for item in results],
            [0, 1, 2],
        )

    def test_empty_documents(self) -> None:
        self.assertEqual(build_ranked_results([], []), [])

    def test_rejects_mismatched_scores(self) -> None:
        with self.assertRaises(ValueError):
            build_ranked_results(["a", "b"], [1.0])

    def test_rejects_non_positive_top_n(self) -> None:
        with self.assertRaises(ValueError):
            build_ranked_results(["a"], [1.0], top_n=0)


if __name__ == "__main__":
    unittest.main()
