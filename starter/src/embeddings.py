import os
import json
import time
import urllib.request
import urllib.error
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingModel:
    """Tạo vector embeddings bằng Google Gemini API với Batching và Exponential Backoff."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        self.batch_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        self.single_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"

    def embed_text(self, text: str) -> list[float]:
        """Tạo embedding cho 1 câu text đơn lẻ (dùng cho truy vấn query)."""
        safe_text = text[:8000].strip()
        if not safe_text:
            return [0.0] * self.dimensions

        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": safe_text}]}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.single_endpoint, data=data, headers={"Content-Type": "application/json"})

        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res.get("embedding", {}).get("values", [])
            except urllib.error.HTTPError as e:
                if e.code in [429, 500, 503]:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"Lỗi embedding sau 5 lần thử lại cho câu: {safe_text[:50]}...")

    def embed_batch(self, texts: list[str], batch_size: int = 30) -> list[list[float]]:
        """Gửi gộp nhiều văn bản (batchEmbedContents) để tăng tốc và tránh quá tải server."""
        all_vectors = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_slice = texts[i:i + batch_size]
            requests_payload = []
            for t in batch_slice:
                safe_t = t[:8000].strip() or " "
                requests_payload.append({
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": safe_t}]}
                })

            payload = {"requests": requests_payload}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.batch_endpoint, data=data, headers={"Content-Type": "application/json"})

            # Retry với exponential backoff khi gặp 503 / 429
            success = False
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        embeddings = res.get("embeddings", [])
                        for item in embeddings:
                            all_vectors.append(item.get("values", []))
                        success = True
                        break
                except urllib.error.HTTPError as e:
                    if e.code in [429, 500, 503]:
                        wait_time = 2 * (attempt + 1)
                        print(f"      [Thông báo] Server Gemini bận (HTTP {e.code}), chờ {wait_time}s rồi thử lại...")
                        time.sleep(wait_time)
                    else:
                        raise
                except Exception as e:
                    time.sleep(2)

            if not success:
                # Fallback: nếu batch lỗi thì gửi từng câu đơn lẻ
                print(f"      [Fallback] Chuyển sang gửi đơn lẻ cho batch {i} -> {i + len(batch_slice)}")
                for t in batch_slice:
                    all_vectors.append(self.embed_text(t))
                    time.sleep(0.1)

            # Nghỉ ngắn giữa các batch
            time.sleep(0.5)

        return all_vectors
