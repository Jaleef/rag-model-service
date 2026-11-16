from sentence_transformers import CrossEncoder


class RerankerModel:

    def __init__(self, model_path: str):
        self.model = CrossEncoder(
            model_path,
            device="cpu",
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 5,
    ):
        pairs = [
            (query, document)
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
        )

        results = [
            {
                "index": index,
                "text": document,
                "score": float(score),
            }
            for index, (document, score)
            in enumerate(zip(documents, scores))
        ]

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_n]

