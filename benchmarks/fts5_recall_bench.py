#!/usr/bin/env python3
"""FTS5 vs BM25 Chinese recall A/B benchmark (M6-7 acceptance).

Corpus: 100 dialogues from extraction_quality_eval.json (as documents).
Queries: each expected fact (should_extract) should recall its source doc.
Metrics: recall@1 / recall@5 / MRR.

Usage: python benchmarks/fts5_recall_bench.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_cases() -> list[dict]:
    data = json.loads(
        (Path(__file__).parent / "extraction_quality_eval.json").read_text(encoding="utf-8")
    )
    return data["items"]


def evaluate(name: str, search_fn, cases: list[dict]) -> dict:
    hits1 = hits5 = mrr = 0.0
    total = 0
    for item in cases:
        doc_id = item["id"]
        for fact in item.get("should_extract", []):
            total += 1
            results = search_fn(fact, 5)
            ids = [r.get("memory_id", "") for r in results]
            if ids[:1] == [doc_id]:
                hits1 += 1
            if doc_id in ids:
                hits5 += 1
                mrr += 1.0 / (ids.index(doc_id) + 1)
    return {
        "engine": name,
        "queries": total,
        "recall@1": round(hits1 / total, 4),
        "recall@5": round(hits5 / total, 4),
        "mrr": round(mrr / total, 4),
    }


def main() -> None:
    from omnimem.memory.unified_index import UnifiedMemoryIndex
    from omnimem.retrieval.bm25 import BM25Retriever
    from omnimem.retrieval.fts5 import FTS5Retriever

    cases = load_cases()

    # Engine A: rank-bm25
    bm25 = BM25Retriever()
    for item in cases:
        bm25.add(item["dialogue"], memory_id=item["id"], metadata={})
    r_bm25 = evaluate("bm25(rank-bm25)", lambda q, k: bm25.search(q, top_k=k), cases)

    # Engine B: unified_index FTS5
    tmp = Path(tempfile.mkdtemp(prefix="fts5bench_"))
    idx = UnifiedMemoryIndex(tmp)
    for item in cases:
        idx.add(
            memory_id=item["id"], wing="w", hall="h", room="r",
            content=item["dialogue"], summary=item["dialogue"][:80],
            content_preview=item["dialogue"][:120],
        )
    idx._write_conn.commit()  # flush batch buffer for read visibility
    fts5 = FTS5Retriever(get_read_conn=lambda: idx._read_conn)
    r_fts5 = evaluate("fts5(unified_index)", lambda q, k: fts5.search(q, top_k=k), cases)

    report = {"dataset": "extraction_quality_eval_v1 dialogues", "results": [r_bm25, r_fts5]}
    out = Path(__file__).parent / "results" / "fts5_vs_bm25_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in (r_bm25, r_fts5):
        print(r)
    verdict = "PASS" if r_fts5["recall@5"] >= r_bm25["recall@5"] else "FAIL"
    print(f"[{verdict}] fts5 recall@5=" + str(r_fts5["recall@5"]) + " vs bm25=" + str(r_bm25["recall@5"]))


if __name__ == "__main__":
    main()
