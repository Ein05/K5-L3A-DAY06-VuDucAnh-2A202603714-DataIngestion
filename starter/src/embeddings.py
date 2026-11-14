from fastembed import TextEmbedding

class LocalFastEmbedModel:
    """Tạo vector embeddings 384 chiều chạy offline 100% bằng FastEmbed."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.dimensions = 384
        self._model = TextEmbedding(model_name=model_name)

    def embed_text(self, text: str) -> list[float]:
        """Tạo embedding cho 1 câu text."""
        return list(self._model.embed([text.strip() or " "]))[0].tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Tạo embedding theo batch cho danh sách văn bản."""
        clean_texts = [t.strip() or " " for t in texts]
        return [v.tolist() for v in self._model.embed(clean_texts, batch_size=batch_size)]
