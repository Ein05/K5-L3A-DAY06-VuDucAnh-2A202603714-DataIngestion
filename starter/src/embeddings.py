from typing import Iterable
from fastembed import TextEmbedding


class LocalFastEmbedModel:
    """Tạo vector embeddings chạy offline 100% bằng ONNX Runtime siêu nhẹ.
    
    Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    - Kích thước: ~470 MB (tối ưu hóa bằng ONNX, không cần PyTorch).
    - Số chiều vector: 384
    - Hỗ trợ tiếng Việt: Xuất sắc, tốc độ xử lý hàng trăm văn bản/giây trên CPU.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        self.dimensions = 384
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_text(self, text: str) -> list[float]:
        """Tạo embedding cho 1 câu text."""
        safe_text = text.strip() or " "
        vec = list(self._model.embed([safe_text]))[0]
        return vec.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Tạo embedding cho danh sách văn bản (tận dụng đa luồng CPU cực nhanh)."""
        clean_texts = [t.strip() or " " for t in texts]
        vectors = []
        # FastEmbed hỗ trợ generator embed tự chia batch
        for vec in self._model.embed(clean_texts, batch_size=batch_size):
            vectors.append(vec.tolist())
        return vectors
