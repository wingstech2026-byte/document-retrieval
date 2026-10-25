#!/usr/bin/env python3
"""Measure retrieval and answer quality on eval/qa_set.json.

    python evaluate.py              run against the database in MONGO_URI / DOCRETRIEVAL_DB
    python evaluate.py --json out.json

Metrics (all computed from the stored passages, none are assumed):
    Recall@k   share of answerable questions with the evidence passage among the top k
    MRR        mean of 1/rank of the first passage that contains the evidence
    nDCG@k     binary relevance: a passage is relevant when it contains the evidence phrase
    doc recall share of the expected documents found in the top k, multi-document questions
    citation   share of cited statements that appear verbatim inside the passage they cite
    answered   share of answerable questions that received a supported answer
    refused    share of unanswerable questions for which the system said it cannot answer
    latency    mean and 95th percentile of ask() in milliseconds
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import docretrieval as dr
import passages as ps

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "qa_set.json")


def _has(text: str, phrase: str) -> bool:
    return ps._norm_ws(phrase).lower() in ps._norm_ws(text).lower()


def run_eval(index: ps.PassageIndex, dataset: dict | None = None, k: int = 5, chat=None, mode: str = "auto") -> dict:
    data = dataset or json.load(open(DATASET, encoding="utf-8"))
    ranks, ndcg, found, answered, cites_ok, cites_all, lat = [], [], [], 0, 0, 0, []
    for item in data["answerable"]:
        r = ps.ask(index, item["q"], k=k, chat=chat, mode=mode)
        lat.append(r["ms"])
        res = index.search(item["q"], k=k, mode=mode)["hits"]
        rel = [h["doc_id"] == item["doc"] and _has(h["text"], item["evidence"]) for h in res]
        first = next((i for i, x in enumerate(rel, start=1) if x), None)
        ranks.append(1 / first if first else 0.0)
        found.append(1.0 if first else 0.0)
        dcg = sum(1 / math.log2(i + 1) for i, x in enumerate(rel, start=1) if x)
        ndcg.append(dcg / (1 / math.log2(2)) if any(rel) else 0.0)       # one relevant passage: ideal DCG = 1
        answered += 1 if r["supported"] else 0
        by_n = {s["n"]: s for s in r["sources"]}
        for seg in r["segments"]:
            for c in seg["citations"]:
                cites_all += 1
                cites_ok += 1 if c in by_n and _has(by_n[c]["text"], seg["text"]) else 0
    doc_recall = []
    for item in data.get("multi_document", []):
        got = {h["doc_id"] for h in index.search(item["q"], k=k * 2, mode=mode)["hits"]}
        doc_recall.append(len(got & set(item["docs"])) / len(item["docs"]))
    refused = 0
    for q in data["unanswerable"]:
        r = ps.ask(index, q, k=k, chat=chat, mode=mode)
        lat.append(r["ms"])
        refused += 0 if r["supported"] else 1
    n_a, n_u = len(data["answerable"]), len(data["unanswerable"])
    lat_sorted = sorted(lat)
    return {
        "k": k, "mode": mode, "answerable": n_a, "unanswerable": n_u, "multi_document": len(doc_recall),
        "recall_at_k": round(sum(found) / n_a, 3), "mrr": round(sum(ranks) / n_a, 3), "ndcg_at_k": round(sum(ndcg) / n_a, 3),
        "doc_recall_multi": round(sum(doc_recall) / len(doc_recall), 3) if doc_recall else None,
        "citation_correct": round(cites_ok / cites_all, 3) if cites_all else None, "citations_checked": cites_all,
        "answered": round(answered / n_a, 3), "refused_unanswerable": round(refused / n_u, 3),
        "latency_ms_mean": round(statistics.mean(lat), 1), "latency_ms_p95": lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=os.environ.get("MONGO_URI", dr.DEFAULT_URI))
    ap.add_argument("--db", default=os.environ.get("DOCRETRIEVAL_DB", dr.DEMO_DB))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", default="auto", choices=["auto", "bm25", "semantic", "hybrid"])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    store = dr.DocumentStore(a.uri, a.db)
    idx = ps.PassageIndex(store, ps.embedder_from_env())
    idx.backfill()
    result = run_eval(idx, k=a.k, chat=ps.chat_from_env(), mode=a.mode)
    for key, value in result.items():
        print(f"{key:22} {value}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
