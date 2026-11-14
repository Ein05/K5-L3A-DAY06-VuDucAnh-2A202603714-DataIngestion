from .models import Document, Chunk


def chunk_strategy_a(documents: list[Document], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Cấu hình A: Kích thước cố định (Fixed-size).
    
    Cắt cố định chunk_size ký tự với khoảng chồng lấn (overlap).
    Metadata: chỉ lưu source và vị trí, không hỗ trợ filter.
    """
    chunks = []
    step = chunk_size - overlap
    for doc in documents:
        text = doc.text
        if len(text) <= chunk_size:
            chunks.append(Chunk(
                chunk_id=f"A_{doc.doc_id}_c0",
                doc_id=doc.doc_id,
                text=text,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("dieu", "Trang 1"),
                    "char_start": 0,
                    "char_end": len(text)
                }
            ))
            continue

        c_idx = 0
        for i in range(0, len(text), step):
            sub_text = text[i:i + chunk_size]
            if not sub_text.strip():
                break
            chunks.append(Chunk(
                chunk_id=f"A_{doc.doc_id}_c{c_idx}",
                doc_id=doc.doc_id,
                text=sub_text,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("dieu", f"Trang {c_idx+1}"),
                    "char_start": i,
                    "char_end": i + len(sub_text)
                }
            ))
            c_idx += 1
            if i + chunk_size >= len(text):
                break
    return chunks


def chunk_strategy_b(documents: list[Document]) -> list[Chunk]:
    """Cấu hình B: Theo đơn vị Điều luật (Semantic / Article-based).
    
    Giữ nguyên vẹn từng Điều luật thành 1 chunk duy nhất nhằm bảo toàn ngữ nghĩa và tính trọn vẹn.
    Metadata: lưu nguồn và tên điều luật, không bật filter.
    """
    chunks = []
    for doc in documents:
        chunks.append(Chunk(
            chunk_id=f"B_{doc.doc_id}",
            doc_id=doc.doc_id,
            text=doc.text,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("dieu", "Điều luật"),
                "title": doc.title
            }
        ))
    return chunks


def chunk_strategy_c(documents: list[Document]) -> list[Chunk]:
    """Cấu hình C: Theo đơn vị Điều luật + Metadata Filter.
    
    Tương tự cấu hình B về mặt phân mảnh văn bản, nhưng gắn kèm các trường metadata hữu ích
    (loai_vb, so_hieu) phục vụ truy vấn lọc trong SQL.
    """
    chunks = []
    for doc in documents:
        chunks.append(Chunk(
            chunk_id=f"C_{doc.doc_id}",
            doc_id=doc.doc_id,
            text=doc.text,
            metadata={
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("dieu", "Điều luật"),
                "title": doc.title,
                "so_hieu": doc.metadata.get("so_hieu", ""),
                "loai_vb": doc.metadata.get("loai_vb", "Văn bản khác")
            }
        ))
    return chunks
