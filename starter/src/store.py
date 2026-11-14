import json
from urllib.parse import urlparse
from typing import Any
import pg8000.native
from .models import Chunk


class PgVectorStore:
    """Lưu trữ và tìm kiếm vector trên PostgreSQL + pgvector sử dụng pg8000."""

    def __init__(self, database_url: str, dimensions: int = 3072) -> None:
        self.dimensions = dimensions
        parsed = urlparse(database_url)
        self.user = parsed.username or "day6"
        self.password = parsed.password or "day6_local_only"
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 55433
        self.database = parsed.path.lstrip("/") or "day6_lab"
        self._get_connection()

    def _get_connection(self) -> pg8000.native.Connection:
        return pg8000.native.Connection(
            user=self.user,
            host=self.host,
            port=self.port,
            database=self.database,
            password=self.password
        )

    def create_schema(self) -> None:
        """Tạo bảng chứa vector và metadata nếu chưa tồn tại."""
        con = self._get_connection()
        try:
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
            con.run("CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);")
        finally:
            con.close()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Lưu danh sách chunk và vector tương ứng vào database.
        
        Sử dụng ON CONFLICT (id) DO UPDATE để không bị nhân đôi bản ghi khi chạy lại.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Số lượng chunks và vectors không khớp nhau")

        con = self._get_connection()
        try:
            for chunk, vec in zip(chunks, vectors):
                # Định dạng vector thành chuỗi '[v1,v2,...]' tương thích kiểu vector của PostgreSQL
                vec_str = "[" + ",".join(str(x) for x in vec) + "]"
                meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
                
                sql = f"""
                    INSERT INTO chunks (id, strategy, doc_id, text, metadata, embedding)
                    VALUES (
                        :id, :strategy, :doc_id, :text, CAST(:metadata AS jsonb), CAST('{vec_str}' AS vector)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                """
                con.run(sql, id=chunk.chunk_id, strategy=strategy, doc_id=chunk.doc_id, text=chunk.text, metadata=meta_json)
        finally:
            con.close()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Truy vấn tìm kiếm Top-K chunk tương đồng nhất theo cosine distance."""
        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        con = self._get_connection()
        try:
            where_clauses = ["strategy = :strategy"]
            params: dict[str, Any] = {"strategy": strategy}

            # Nếu có filter metadata (dùng cho chiến lược C)
            if filters:
                for k, v in filters.items():
                    param_name = f"flt_{k}"
                    where_clauses.append(f"metadata->>'{k}' = :{param_name}")
                    params[param_name] = str(v)

            where_sql = " AND ".join(where_clauses)
            sql = f"""
                SELECT id, doc_id, text, metadata, 1 - (embedding <=> CAST('{vec_str}' AS vector)) AS score
                FROM chunks
                WHERE {where_sql}
                ORDER BY embedding <=> CAST('{vec_str}' AS vector) ASC
                LIMIT {top_k};
            """

            rows = con.run(sql, **params)
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "doc_id": r[1],
                    "text": r[2],
                    "metadata": r[3],
                    "score": float(r[4])
                })
            return results
        finally:
            con.close()
