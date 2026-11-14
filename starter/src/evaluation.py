from typing import Any

def evaluate_retrieval(retrieved: list[dict[str, Any]], gold_doc_ids: list[str]) -> dict[str, bool]:
    """Kiểm tra xem Top-1 và Top-3 có chứa gold document không."""
    gold_set = set(gold_doc_ids)
    return {
        "hit@1": bool(retrieved and retrieved[0]["doc_id"] in gold_set),
        "hit@3": any(r["doc_id"] in gold_set for r in retrieved[:3])
    }

def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tính tỷ lệ phần trăm và phân số Hit@1, Hit@3."""
    n = len(records)
    h1 = sum(1 for r in records if r["hit@1"])
    h3 = sum(1 for r in records if r["hit@3"])
    return {
        "hit@1": round(h1 / n, 4) if n else 0,
        "hit@3": round(h3 / n, 4) if n else 0,
        "hit@1_count": f"{h1}/{n}",
        "hit@3_count": f"{h3}/{n}"
    }
