from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Đại diện cho một văn bản/điều luật gốc."""
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Đại diện cho một đoạn văn bản sau khi chia nhỏ (chunking)."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
