from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_path: str):
        self.model = SentenceTransformer(
            model_path,
            device="cpu",
        )

    def encode(self, texts: list[str], input_type: str = "document"):
        if input_type == "query":
            embeddings = self.model.encode_query(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            embeddings = self.model.encode_document(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        return embeddings.tolist()

