"""Scale benchmark for the report: loads synthetic documents into a throw-away database and times each function.

    python docs/benchmark.py [number_of_documents]        (default 10000)

Writes docs/benchmark_results.json. The synthetic corpus reuses the vocabulary of the sample documents with a skewed (Zipf-like)
word distribution, so common and rare terms both occur.
"""
import json
import os
import platform
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import docretrieval as dr  # noqa: E402
import sample_data  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
rnd = random.Random(2026)
samples = sample_data.documents()
vocab = sorted({t for d in samples for t in dr.tokenize(d["text"] + " " + d["title"])})
weights = [1.0 / (i + 1) ** 0.9 for i in range(len(vocab))]
rnd.shuffle(vocab)  # the rank-frequency order is random, not alphabetical
authors = [f"Author {chr(65 + i % 26)}{i // 26}" for i in range(300)]
keywords = [f"topic {w}" for w in vocab[:120]]
types = list(dr.DOC_TYPES)


def make_doc(i):
    words = rnd.choices(vocab, weights, k=rnd.randint(110, 190))
    text = " ".join(words)
    return dict(doc_id=f"b{i:06d}", doc_type=types[i % 4], year=2010 + i % 16, title=" ".join(rnd.choices(vocab, weights, k=6)).title(),
                authors=rnd.sample(authors, rnd.randint(1, 3)), keywords=rnd.sample(keywords, rnd.randint(3, 5)),
                abstract=" ".join(words[:25]), text=text, file_bytes=text.encode(), filename=f"b{i:06d}.txt")


def ms(t0):
    return (time.perf_counter() - t0) * 1000


def summary(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 2), "p95_ms": round(xs[int(len(xs) * 0.95) - 1], 2), "max_ms": round(xs[-1], 2)}


db_name = "docretrieval_bench"
st = dr.DocumentStore(db_name=db_name)
st.drop_database()
st = dr.DocumentStore(db_name=db_name)
res = {"documents": N, "python": platform.python_version(), "os": platform.platform(), "mongo": st.client.server_info()["version"]}

# ---- ingest -------------------------------------------------------------------------
t0 = time.perf_counter()
lat = []
for i in range(N):
    d = make_doc(i)
    t = time.perf_counter()
    st.add_document(**d)
    lat.append(ms(t))
    if (i + 1) % 2000 == 0:
        print(f"  added {i + 1}", flush=True)
total = time.perf_counter() - t0
res["ingest"] = {"total_s": round(total, 1), "docs_per_s": round(N / total, 1), **{k: v for k, v in summary(lat).items() if k != "n"}}
st.client.admin.command("fsync")  # checkpoint, so the size statistics are current
stats = st.stats()
dbs = st.db.command("dbStats")
res["size"] = {"postings": stats["postings"], "distinct_terms": stats["distinct_terms"], "gridfs_files": stats["gridfs_files"],
               "gridfs_mb": round(stats["gridfs_bytes"] / 1e6, 1), "data_mb": round(dbs["dataSize"] / 1e6, 1),
               "index_mb": round(dbs["indexSize"] / 1e6, 1),
               "avg_doc_length": stats["avg_doc_length"]}
print("ingest done", res["ingest"], res["size"], flush=True)

# ---- queries ---------------------------------------------------------------------------
# pick query terms by their real document frequency (df): frequent = the 10 most widespread terms, medium = around the median, rare = lowest
dfs = sorted(st._vocabulary().items(), key=lambda kv: -kv[1])
frequent = [t for t, _ in dfs[:10]]
mid = [t for t, _ in dfs[len(dfs) // 2 - 10: len(dfs) // 2 + 10]]
rare = [t for t, _ in dfs[-150:]]
res["term_df"] = {"frequent_df": [dfs[0][1], dfs[9][1]], "medium_df": [dfs[len(dfs) // 2 + 10][1], dfs[len(dfs) // 2 - 10][1]], "rare_df": [dfs[-1][1], dfs[-150][1]]}
def q2(group): return " ".join(rnd.sample(group, 2))
def q_mixed(): return rnd.choice(frequent) + " " + rnd.choice(rare) + " " + rnd.choice(rare)
cases = {
    "search, 2 frequent terms": lambda: st.search(q2(frequent), limit=10),
    "search, 2 medium terms": lambda: st.search(q2(mid), limit=10),
    "search, 2 rare terms": lambda: st.search(q2(rare), limit=10),
    "search, 1 frequent + 2 rare terms": lambda: st.search(q_mixed(), limit=10),
    "search with type and year filters": lambda: st.search(q_mixed(), doc_type="patent", year_from=2015, year_to=2022, limit=10),
    "search with author filter": lambda: st.search(q_mixed(), author="Author B", limit=10),
    "phrase search (2 medium terms)": lambda: st.search('"' + q2(mid) + '"', limit=10),
}
res["queries"] = {}
for name, fn in cases.items():
    for _ in range(5):
        fn()  # warm up
    xs = []
    for _ in range(100):
        t = time.perf_counter()
        fn()
        xs.append(ms(t))
    res["queries"][name] = summary(xs)
    print(name, res["queries"][name], flush=True)

ids = [f"b{rnd.randrange(N):06d}" for _ in range(40)]
def timed(fn, items):
    xs = []
    for x in items:
        t = time.perf_counter()
        fn(x)
        xs.append(ms(t))
    return summary(xs)
res["other"] = {
    "related documents": timed(lambda i: st.related(i, 5), ids[:30]),
    "compare two documents": timed(lambda i: st.compare(i, ids[0]), ids[1:31]),
    "read one page": timed(lambda i: st.read(i, 1), ids),
    "download (GridFS + SHA-256)": timed(lambda i: st.download(i, os.path.join(ROOT, "downloads_bench")), ids[:30]),
    "keyword co-occurrence (whole collection)": timed(lambda _: st.keyword_cooccurrence(min_count=2, limit=50), range(5)),
    "statistics": timed(lambda _: st.stats(), range(3)),
}
for k, v in res["other"].items():
    print(k, v, flush=True)

# ---- what the index buys: one term lookup with and without the (term, doc_id) index ---------------------
term = frequent[0]
def lookup(hint=None):
    cur = st.postings.find({"term": term}, {"_id": 0})
    if hint:
        cur = cur.hint(hint)
    return len(list(cur))
for _ in range(3):
    lookup()
with_idx = []
scan = []
for _ in range(20):
    t = time.perf_counter(); n_idx = lookup(); with_idx.append(ms(t))
    t = time.perf_counter(); n_scan = lookup([("$natural", 1)]); scan.append(ms(t))
plan = st.postings.find({"term": frequent[1]}).explain()["queryPlanner"]["winningPlan"]
plan_s = json.dumps(plan)
res["index_effect"] = {"term": term, "matching_postings": n_idx, "with_index_ms": round(statistics.median(with_idx), 2),
                       "collection_scan_ms": round(statistics.median(scan), 2), "same_result": n_idx == n_scan,
                       "plan_uses_index": "IXSCAN" in plan_s, "total_postings": stats["postings"]}
rare_term = rare[len(rare) // 2]
t = time.perf_counter(); n_r = len(list(st.postings.find({"term": rare_term}))); r_idx = ms(t)
t = time.perf_counter(); list(st.postings.find({"term": rare_term}).hint([("$natural", 1)])); r_scan = ms(t)
res["index_effect"].update({"rare_term_postings": n_r, "rare_with_index_ms": round(r_idx, 2), "rare_collection_scan_ms": round(r_scan, 2)})
print("index effect", res["index_effect"], flush=True)

# ---- delete ------------------------------------------------------------------------------------------------
dels = [f"b{i:06d}" for i in range(N - 100, N)]
before = st.stats()
res["delete"] = timed(lambda i: st.delete_document(i), dels)
after = st.stats()
res["delete"].update({"postings_before": before["postings"], "postings_after": after["postings"], "files_before": before["gridfs_files"],
                      "files_after": after["gridfs_files"], "orphans": st.postings.count_documents({"doc_id": {"$in": dels}})})
print("delete", res["delete"], flush=True)

st.drop_database()
json.dump(res, open(os.path.join(ROOT, "docs", f"benchmark_results_{N}.json"), "w"), indent=1)
import shutil
shutil.rmtree(os.path.join(ROOT, "downloads_bench"), ignore_errors=True)
print("done")
