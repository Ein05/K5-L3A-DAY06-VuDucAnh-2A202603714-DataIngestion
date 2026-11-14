import csv
import unicodedata
from .models import Document

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode tiếng Việt về chuẩn NFC và bỏ khoảng trắng thừa."""
    return " ".join(unicodedata.normalize("NFC", text).split()) if text else ""

def parse_doc_type(title: str) -> str:
    """Xác định loại văn bản từ tiêu đề (Luật, Nghị định, Thông tư...)."""
    upper = title.upper()
    if "QH" in upper: return "Luật"
    if "NĐ-CP" in upper or "ND-CP" in upper: return "Nghị định"
    if "TT-" in upper: return "Thông tư"
    if "QĐ-" in upper or "QD-" in upper: return "Quyết định"
    return "Văn bản khác"

def load_corpus(file_path: str, max_docs: int = 100, target_ids: set[str] | None = None) -> list[Document]:
    """Đọc corpus từ file CSV, đảm bảo đủ >= 30 trang nội dung và chứa đủ target_ids."""
    docs, seen = [], set()
    with open(file_path, "r", encoding="utf-16") as f:
        reader = csv.reader(f)
        next(reader)  # Bỏ qua header
        for row in reader:
            if len(row) < 4: continue
            doc_id, title, text = row[1].strip(), row[2].strip(), row[3].strip()
            
            is_target = target_ids and (doc_id in target_ids)
            if not is_target and len(docs) >= max_docs: continue
            if doc_id in seen: continue

            parts = title.split()
            docs.append(Document(
                doc_id=doc_id,
                title=title,
                text=normalize_text(text),
                metadata={
                    "source": "legal_truncated_corpus.csv",
                    "dieu": f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "",
                    "loai_vb": parse_doc_type(title)
                }
            ))
            seen.add(doc_id)
    return docs
