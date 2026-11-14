import os
import json
import time
from .ingestion import load_corpus
from .chunking import chunk_strategy_a, chunk_strategy_b, chunk_strategy_c
from .embeddings import LocalFastEmbedModel
from .store import PgVectorStore
from .evaluation import evaluate_retrieval, compute_metrics

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(root)

    # 1. Đọc 20 câu hỏi benchmark
    with open(os.path.join(repo, "benchmark", "questions.json"), "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    dev_q, test_q = benchmark["dev"], benchmark["test"]
    all_q = dev_q + test_q
    target_ids = {gid for q in all_q for gid in q.get("gold_doc_ids", [])}

    # 2. Ingest corpus (> 30 trang)
    corpus_path = os.path.join(root, "data", "legal_truncated_corpus.csv")
    docs = load_corpus(corpus_path, max_docs=100, target_ids=target_ids)
    print(f"[1] Ingested: {len(docs)} docs, {sum(len(d.text) for d in docs):,} chars (~{sum(len(d.text) for d in docs)/1500:.1f} pages)")

    # 3. Phân mảnh theo 3 chiến lược
    strategies = {
        "A": chunk_strategy_a(docs, chunk_size=500, overlap=50),
        "B": chunk_strategy_b(docs),
        "C": chunk_strategy_c(docs)
    }

    # 4. Khởi tạo Embedder và Vector Store
    db_url = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
    embedder = LocalFastEmbedModel()
    store = PgVectorStore(db_url, dimensions=embedder.dimensions)
    store.create_schema()

    # 5. Sinh vector và Lưu vào pgvector
    for name, chunks in strategies.items():
        t0 = time.time()
        vectors = embedder.embed_batch([c.text for c in chunks])
        store.upsert(name, chunks, vectors)
        print(f"[2] Strategy {name}: {len(chunks)} chunks embedded & saved in {time.time()-t0:.2f}s")

    # 6. Chạy Benchmark và Đánh giá
    print("\n[3] EVALUATION RESULTS (Hit@1 and Hit@3):")
    print(f"{'Strategy':<10} | {'Dev Hit@1':<10} | {'Dev Hit@3':<10} | {'Test Hit@1':<10} | {'Test Hit@3':<10}")
    print("-" * 55)

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    for name in ["A", "B", "C"]:
        dev_res, test_res, trace = [], [], []
        for q in all_q:
            flt = q.get("filter") if name == "C" else None
            top3 = store.search(name, embedder.embed_text(q["question"]), top_k=3, filters=flt)
            ev = evaluate_retrieval(top3, q["gold_doc_ids"])
            
            trace.append({**q, **ev, "retrieved_top3": top3})
            (dev_res if any(d["id"] == q["id"] for d in dev_q) else test_res).append(ev)

        with open(os.path.join(results_dir, f"trace_strategy_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

        dm, tm = compute_metrics(dev_res), compute_metrics(test_res)
        print(f"Strategy {name:<2} | {dm['hit@1_count']:<10} | {dm['hit@3_count']:<10} | {tm['hit@1_count']:<10} | {tm['hit@3_count']:<10}")

if __name__ == "__main__":
    main()
