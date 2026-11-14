import json
from urllib.parse import urlparse
from typing import Any
import pg8000.native
from .models import Chunk

class PgVectorStore:
    """Lưu trữ và tìm kiếm vector trên PostgreSQL + pgvector qua driver pg8000."""

    def __init__(self, database_url: str, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        u = urlparse(database_url)
        self.conn_args = {
            "user": u.username or "day6",
            "password": u.password or "day6_local_only",
            "host": u.hostname or "127.0.0.1",
            "port": u.port or 55433,
            "database": u.path.lstrip("/") or "day6_lab"
        }

    def _connect(self) -> pg8000.native.Connection:
        return pg8000.native.Connection(**self.conn_args)

    def create_schema(self) -> None:
        """Tạo bảng chunks có cột vector nếu chưa có."""
        con = self._connect()
        con.run("CREATE EXTENSION IF NOT EXISTS vector;")
        con.run(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL,
                embedding vector({self.dimensions})
            );
        """)
        con.close()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Lưu chunks và vectors vào DB, không bị trùng bản ghi khi chạy lại."""
        con = self._connect()
        for chunk, vec in zip(chunks, vectors):
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            con.run(f"""
                INSERT INTO chunks (id, strategy, doc_id, text, metadata, embedding)
                VALUES (:id, :strategy, :doc_id, :text, CAST(:meta AS jsonb), CAST('{vec_str}' AS vector))
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding;
            """, id=chunk.chunk_id, strategy=strategy, doc_id=chunk.doc_id, text=chunk.text, meta=json.dumps(chunk.metadata))
        con.close()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Truy vấn Top-K chunk gần nhất theo Cosine Distance."""
        con = self._connect()
        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        flt_sql = f"AND metadata->>'loai_vb' = '{filters['loai_vb']}'" if filters and "loai_vb" in filters else ""
        
        rows = con.run(f"""
            SELECT id, doc_id, text, metadata, 1 - (embedding <=> CAST('{vec_str}' AS vector)) AS score
            FROM chunks
            WHERE strategy = '{strategy}' {flt_sql}
            ORDER BY embedding <=> CAST('{vec_str}' AS vector) ASC
            LIMIT {top_k};
        """)
        con.close()
        return [{"id": r[0], "doc_id": r[1], "text": r[2], "metadata": r[3], "score": float(r[4])} for r in rows]
