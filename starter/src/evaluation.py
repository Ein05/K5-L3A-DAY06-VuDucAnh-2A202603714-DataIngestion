from typing import Any


def evaluate_retrieval(query_results: list[dict[str, Any]], gold_doc_ids: list[str]) -> dict[str, bool]:
    """Đánh giá Hit@1 và Hit@3 cho một câu truy vấn."""
    if not query_results:
        return {"hit@1": False, "hit@3": False}

    gold_set = set(gold_doc_ids)

    # Hit@1: Chunk xếp hạng 1 có khớp với gold doc không
    top_1_hit = query_results[0].get("doc_id") in gold_set

    # Hit@3: Có ít nhất một chunk trong Top 3 khớp với gold doc không
    top_3_hit = any(res.get("doc_id") in gold_set for res in query_results[:3])

    return {
        "hit@1": top_1_hit,
        "hit@3": top_3_hit
    }


def compute_metrics(eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tổng hợp chỉ số Hit@1 và Hit@3 trên toàn bộ tập câu hỏi."""
    total = len(eval_records)
    if total == 0:
        return {"total": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@1_count": 0, "hit@3_count": 0}

    hit1_count = sum(1 for r in eval_records if r.get("hit@1"))
    hit3_count = sum(1 for r in eval_records if r.get("hit@3"))

    return {
        "total": total,
        "hit@1": round(hit1_count / total, 4),
        "hit@3": round(hit3_count / total, 4),
        "hit@1_count": f"{hit1_count}/{total}",
        "hit@3_count": f"{hit3_count}/{total}"
    }
