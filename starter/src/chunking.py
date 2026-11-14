from .models import Document, Chunk

def chunk_strategy_a(documents: list[Document], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Chiến lược A: Cắt cố định 500 ký tự (Fixed-size)."""
    chunks, step = [], chunk_size - overlap
    for doc in documents:
        text = doc.text
        if len(text) <= chunk_size:
            chunks.append(Chunk(f"A_{doc.doc_id}_c0", doc.doc_id, text, {"source": "legal", "dieu": doc.metadata["dieu"]}))
            continue
        for idx, i in enumerate(range(0, len(text), step)):
            sub = text[i:i + chunk_size]
            if not sub.strip(): break
            chunks.append(Chunk(f"A_{doc.doc_id}_c{idx}", doc.doc_id, sub, {"source": "legal", "dieu": doc.metadata["dieu"]}))
            if i + chunk_size >= len(text): break
    return chunks

def chunk_strategy_b(documents: list[Document]) -> list[Chunk]:
    """Chiến lược B: Cắt theo từng Điều luật trọn vẹn (Semantic Article-based)."""
    return [
        Chunk(f"B_{d.doc_id}", d.doc_id, d.text, {"source": "legal", "dieu": d.metadata["dieu"]})
        for d in documents
    ]

def chunk_strategy_c(documents: list[Document]) -> list[Chunk]:
    """Chiến lược C: Theo Điều luật + Metadata Filter (loai_vb)."""
    return [
        Chunk(f"C_{d.doc_id}", d.doc_id, d.text, {
            "source": "legal",
            "dieu": d.metadata["dieu"],
            "loai_vb": d.metadata["loai_vb"]
        })
        for d in documents
    ]
