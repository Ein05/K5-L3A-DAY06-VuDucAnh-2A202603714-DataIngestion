import os
import json
import time
from dotenv import load_dotenv

from .ingestion import load_corpus
from .chunking import chunk_strategy_a, chunk_strategy_b, chunk_strategy_c
from .embeddings import GeminiEmbeddingModel
from .store import PgVectorStore
from .evaluation import evaluate_retrieval, compute_metrics

load_dotenv()


def main():
    print("=" * 60)
    print("BẮT ĐẦU CHẠY PIPELINE DATA INGESTION & BENCHMARK")
    print("=" * 60)

    # 1. Tải benchmark questions
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(base_dir)
    benchmark_file = os.path.join(repo_root, "benchmark", "questions.json")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    dev_questions = benchmark["dev"]
    test_questions = benchmark["test"]
    all_questions = dev_questions + test_questions

    # Tập hợp các gold_doc_ids cần thiết
    target_doc_ids = set()
    for q in all_questions:
        for gid in q.get("gold_doc_ids", []):
            target_doc_ids.add(gid)

    # 2. Đọc và chuẩn hóa dữ liệu
    corpus_file = os.path.join(base_dir, "data", "legal_truncated_corpus.csv")
    print(f"\n[1/5] Đang đọc tài liệu từ {corpus_file}...")
    documents = load_corpus(corpus_file, max_docs=100, target_doc_ids=target_doc_ids)
    
    total_chars = sum(len(d.text) for d in documents)
    print(f"-> Đã tải thành công {len(documents)} văn bản/điều luật.")
    print(f"-> Tổng ký tự: {total_chars:,} ký tự (~ {total_chars / 1500:.1f} trang quy chuẩn, đạt yêu cầu >=30 trang).")

    # 3. Phân mảnh theo 3 chiến lược
    print("\n[2/5] Thực hiện phân mảnh dữ liệu:")
    chunks_a = chunk_strategy_a(documents, chunk_size=500, overlap=50)
    chunks_b = chunk_strategy_b(documents)
    chunks_c = chunk_strategy_c(documents)

    print(f"-> Chiến lược A (Fixed-size 500 ký tự): {len(chunks_a)} chunks")
    print(f"-> Chiến lược B (Theo Điều luật):        {len(chunks_b)} chunks")
    print(f"-> Chiến lược C (Điều luật + Metadata):  {len(chunks_c)} chunks")

    # 4. Khởi tạo DB và Embedding Model
    db_url = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
    embedder = GeminiEmbeddingModel()
    store = PgVectorStore(database_url=db_url, dimensions=embedder.dimensions)
    store.create_schema()

    # 5. Sinh vector và Upsert vào pgvector cho từng chiến lược
    strategies = [
        ("A", chunks_a),
        ("B", chunks_b),
        ("C", chunks_c)
    ]

    print("\n[3/5] Tạo Vector Embeddings & Lưu vào PostgreSQL + pgvector:")
    for strat_name, chunks in strategies:
        print(f"-> Đang xử lý Chiến lược {strat_name} ({len(chunks)} chunks)...")
        texts = [c.text for c in chunks]
        t0 = time.time()
        vectors = embedder.embed_batch(texts, batch_size=30)
        store.upsert(strategy=strat_name, chunks=chunks, vectors=vectors)
        t_elapsed = round(time.time() - t0, 2)
        print(f"   Hoàn thành chiến lược {strat_name} trong {t_elapsed}s.")

    # 6. Chạy Benchmark cho cả 3 chiến lược
    print("\n[4/5] Chạy đánh giá Benchmark trên 20 câu hỏi (15 Dev / 5 Test):")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_results = {}

    for strat_name, _ in strategies:
        trace_records = []
        dev_records = []
        test_records = []

        for q in all_questions:
            q_id = q["id"]
            question_text = q["question"]
            gold_docs = q["gold_doc_ids"]
            
            # Chỉ áp dụng metadata filter ở chiến lược C
            flt = q.get("filter") if strat_name == "C" else None

            # Tạo query vector
            q_vec = embedder.embed_text(question_text)
            top_results = store.search(strategy=strat_name, query_vector=q_vec, top_k=3, filters=flt)

            eval_res = evaluate_retrieval(top_results, gold_docs)

            record = {
                "question_id": q_id,
                "question": question_text,
                "filter_applied": flt,
                "gold_doc_ids": gold_docs,
                "hit@1": eval_res["hit@1"],
                "hit@3": eval_res["hit@3"],
                "retrieved_top3": top_results
            }
            trace_records.append(record)

            # Phân loại dev / test
            is_dev = any(dq["id"] == q_id for dq in dev_questions)
            if is_dev:
                dev_records.append(eval_res)
            else:
                test_records.append(eval_res)

        # Lưu trace chi tiết
        trace_file = os.path.join(results_dir, f"trace_strategy_{strat_name}.json")
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_records, f, ensure_ascii=False, indent=2)

        dev_metrics = compute_metrics(dev_records)
        test_metrics = compute_metrics(test_records)

        summary_results[strat_name] = {
            "dev": dev_metrics,
            "test": test_metrics
        }

    # 7. In bảng kết quả tổng hợp
    print("\n" + "=" * 60)
    print("KẾT QUẢ BENCHMARK TRUY XUẤT (Hit@1 và Hit@3)")
    print("=" * 60)
    print(f"{'Chiến lược':<15} | {'Dev Hit@1':<12} | {'Dev Hit@3':<12} | {'Test Hit@1':<12} | {'Test Hit@3':<12}")
    print("-" * 75)
    for s in ["A", "B", "C"]:
        dev_m = summary_results[s]["dev"]
        test_m = summary_results[s]["test"]
        print(f"Chiến lược {s:<5} | {dev_m['hit@1_count']:<12} | {dev_m['hit@3_count']:<12} | {test_m['hit@1_count']:<12} | {test_m['hit@3_count']:<12}")
    print("=" * 60)

    # Lưu kết quả tổng quan
    metrics_file = os.path.join(results_dir, "metrics_summary.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    print(f"\n[5/5] Hoàn tất! Toàn bộ trace và chỉ số đã được lưu tại thư mục `starter/results/`.")


if __name__ == "__main__":
    main()
