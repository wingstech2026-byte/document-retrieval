"""Tests for the document retrieval system. They need a real MongoDB
(default mongodb://localhost:27017, override with $MONGO_URI) and use a
throw-away database that is dropped afterwards."""
import hashlib
import math
import os
import uuid
from collections import Counter
from itertools import combinations

import pytest
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

import docretrieval as dr
import sample_data

URI = os.environ.get("MONGO_URI", dr.DEFAULT_URI)


@pytest.fixture(scope="session")
def mongo_client():
    client = MongoClient(URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        pytest.skip(f"no MongoDB reachable at {URI}")
    yield client
    client.close()


@pytest.fixture
def empty_store(mongo_client):
    store = dr.DocumentStore(client=mongo_client, db_name=f"docretrieval_test_{uuid.uuid4().hex[:10]}")
    yield store
    store.drop_database()


@pytest.fixture
def store(empty_store):
    for d in sample_data.documents():
        empty_store.add_document(**d)
    return empty_store


def ids(results):
    return [r["doc_id"] for r in results]


# --------------------------------------------------------------------------
# Storage and index structure
# --------------------------------------------------------------------------
def test_add_creates_metadata_postings_and_gridfs_file(store):
    doc = store.get_document("pat-001")
    assert doc["type"] == "patent" and doc["year"] == 2020
    assert doc["authors"] == ["Sofia Alvarez"]
    assert "inverted index" in doc["keywords"]
    assert store.postings.count_documents({"doc_id": "pat-001"}) > 20
    assert store.db["fs.files"].count_documents({"_id": doc["file_id"]}) == 1
    n, total = store._corpus()
    assert n == 10 and total == sum(d["length"] for d in store.documents.find())


def test_postings_have_unique_term_doc_index(store):
    sample = store.postings.find_one({"doc_id": "pap-001"})
    with pytest.raises(DuplicateKeyError):
        store.postings.insert_one({"term": sample["term"], "doc_id": "pap-001", "tf": 99})
    idx = store.postings.index_information()
    assert any(v.get("unique") and [k for k, _ in v["key"]] == ["term", "doc_id"] for v in idx.values())


def test_all_four_document_types_are_supported(store):
    assert {d["type"] for d in store.documents.find()} == set(dr.DOC_TYPES)
    assert dr._norm_type("Technical Report") == "report"


def test_invalid_input_writes_nothing(empty_store):
    with pytest.raises(dr.DocError):
        empty_store.add_document("Bad type", "novel", text="x")
    with pytest.raises(dr.DocError):
        empty_store.add_document("   ", "paper", text="x")
    with pytest.raises(dr.DocError):
        empty_store.add_document("Nope", "paper", file_path="does/not/exist.pdf")
    assert empty_store.documents.count_documents({}) == 0
    assert empty_store.postings.count_documents({}) == 0
    assert empty_store.db["fs.files"].count_documents({}) == 0


def test_duplicate_id_is_rejected_without_side_effects(store):
    before = store.postings.count_documents({})
    with pytest.raises(dr.DocError):
        store.add_document("Again", "paper", text="alpha beta", doc_id="pap-001", file_bytes=b"zz", filename="z.txt")
    assert store.postings.count_documents({}) == before
    assert store.db["fs.files"].count_documents({}) == 10


def test_text_is_extracted_from_text_files(empty_store, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Zirconium crucible thermal shock test results.", encoding="utf-8")
    r = empty_store.add_document("Crucible note", "report", file_path=str(f))
    assert "zirconium" in empty_store.get_document(r["doc_id"])["text"].lower()
    assert ids(empty_store.search("zirconium")) == [r["doc_id"]]


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def test_bm25_matches_hand_computed_scores(empty_store):
    s = empty_store
    s.add_document("Doc one", "paper", text="zebra zebra zebra filler filler", doc_id="a")
    s.add_document("Doc two", "paper", text="zebra filler filler filler filler filler filler", doc_id="b")
    s.add_document("Doc three", "paper", text="quartz filler filler", doc_id="c")
    res = s.search("zebra", snippet=False)
    assert ids(res) == ["a", "b"]  # more occurrences and a shorter document win

    n, total = s._corpus()
    avgdl = total / n
    df = s.postings.count_documents({"term": "zebra"})
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    for r in res:
        tf = s.postings.find_one({"term": "zebra", "doc_id": r["doc_id"]})["tf"]
        dl = s.get_document(r["doc_id"])["length"]
        expected = idf * tf * (dr.K1 + 1) / (tf + dr.K1 * (1 - dr.B + dr.B * dl / avgdl))
        assert r["score"] == pytest.approx(expected, abs=1e-4)


def test_rare_terms_outweigh_common_terms(store):
    # "index" is spread over several documents, "compression" appears in one.
    res = store.search("index compression", snippet=False)
    assert res[0]["doc_id"] == "pat-001"
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_title_and_keywords_are_searchable(store):
    # "failover" and "replica" are in the title; ranking must put that report first.
    assert store.search("replica failover")[0]["doc_id"] == "rep-001"
    # "technical drawing" is only a keyword of drw-001.
    assert store.search("technical drawing", snippet=False)[0]["doc_id"] == "drw-001"


def test_plural_and_case_insensitive_matching(store):
    assert ids(store.search("GEARBOXES", snippet=False)) == ids(store.search("gearbox", snippet=False))
    assert store.search("gearbox", snippet=False)


def test_unknown_and_stopword_only_queries_return_nothing(store):
    assert store.search("xyzzyplugh") == []
    assert store.search("the of and") == []
    assert store.search("") == []


def test_require_all_terms(store):
    any_hits = ids(store.search("bearing failover", snippet=False))
    all_hits = ids(store.search("bearing failover", require_all=True, snippet=False))
    assert set(any_hits) >= {"rep-001", "rep-002"} and all_hits == []
    with_both = set(store.postings.distinct("doc_id", {"term": "bearing"})) & set(
        store.postings.distinct("doc_id", {"term": "vibration"}))
    assert with_both == {"rep-002", "drw-001", "pap-003"}
    assert set(ids(store.search("bearing vibration", require_all=True, snippet=False))) == with_both


def test_snippet_highlights_query_terms(store):
    top = store.search("posting list")[0]
    assert top["doc_id"] == "pat-001"
    assert "**posting**" in top["snippet"].lower()
    assert len(top["snippet"].split()) <= 40


def test_limit(store):
    assert len(store.search("document", limit=2, snippet=False)) == 2


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def test_filter_by_type(store):
    res = store.search("vibration", snippet=False)
    assert {r["type"] for r in res} >= {"patent", "paper", "report"}
    only = store.search("vibration", doc_type="patent", snippet=False)
    assert ids(only) == ["pat-002"]
    assert ids(store.search("vibration", doc_type="technical report", snippet=False)) == ["rep-002"]


def test_filter_by_year_range(store):
    res = store.search("vibration", year_from=2023, snippet=False)
    assert res and all(r["year"] >= 2023 for r in res)
    res = store.search("vibration", year_to=2022, snippet=False)
    assert ids(res) == ["pat-002"]
    res = store.search("vibration", year_from=2022, year_to=2023, snippet=False)
    assert all(2022 <= r["year"] <= 2023 for r in res)
    assert store.search("vibration", year_from=1990, year_to=1999) == []


def test_filter_by_author_is_case_insensitive_substring(store):
    res = store.search("sensor vibration", author="tomas berg", snippet=False)
    assert res and all("Tomas Berg" in r["authors"] for r in res)
    assert ids(store.search("index", author="ALVAREZ", snippet=False)) == ids(
        [r for r in store.search("index", snippet=False) if "Sofia Alvarez" in r["authors"]])
    assert store.search("index", author="nobody at all") == []


def test_filters_combine(store):
    res = store.search("nosql cluster", doc_type="report", year_from=2023, author="weber", snippet=False)
    assert ids(res) == ["rep-001"]
    assert store.search("nosql cluster", doc_type="patent", author="weber") == []


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def test_reading_is_paginated_and_lossless(store):
    text = store.get_document("pap-001")["text"]
    first = store.read("pap-001", page=1, page_size=400)
    assert first["pages"] > 2
    pages = [store.read("pap-001", page=p, page_size=400)["content"] for p in range(1, first["pages"] + 1)]
    assert "".join(pages) == text
    assert all(len(p) <= 400 for p in pages)


def test_reading_out_of_range_page_fails(store):
    pages = store.read("pap-001", page_size=400)["pages"]
    for bad in (0, pages + 1):
        with pytest.raises(dr.DocError):
            store.read("pap-001", page=bad, page_size=400)
    with pytest.raises(dr.NotFound):
        store.read("nope-000")


def test_drawing_can_be_read_as_its_description(store):
    r = store.read("drw-001")
    assert r["pages"] == 1 and "bearing bores" in r["content"] and r["has_file"]


def test_split_pages_edge_cases():
    assert dr.split_pages("", 100) == [""]
    text = "word " * 500
    assert "".join(dr.split_pages(text, 60)) == text


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def test_compare_identical_document_is_one(store):
    c = store.compare("pap-001", "pap-001")
    assert c["cosine"] == pytest.approx(1.0, abs=1e-4)
    assert c["only_a"] == [] and c["only_b"] == []
    assert c["metadata"]["author_jaccard"] == 1.0 and c["metadata"]["keyword_jaccard"] == 1.0


def test_compare_is_symmetric_and_bounded(store):
    ab, ba = store.compare("pap-003", "rep-002"), store.compare("rep-002", "pap-003")
    assert ab["cosine"] == ba["cosine"]
    assert [t for t, _ in ab["shared_terms"]] == [t for t, _ in ba["shared_terms"]]
    assert [t for t, _ in ab["only_a"]] == [t for t, _ in ba["only_b"]]
    assert 0 <= ab["cosine"] <= 1


def test_related_topics_score_higher_than_unrelated(store):
    close = store.compare("pap-003", "rep-002")["cosine"]
    far = store.compare("pap-003", "pat-001")["cosine"]
    assert close > 3 * far


def test_compare_reports_shared_and_distinctive_terms(store):
    c = store.compare("pap-003", "rep-002")
    shared = {t for t, _ in c["shared_terms"]}
    assert {"gearbox", "vibration", "fault"} <= shared
    assert "federated" in {t for t, _ in c["only_a"]}
    assert "baseline" in {t for t, _ in c["only_b"]}
    assert not shared & {t for t, _ in c["only_a"]} and not shared & {t for t, _ in c["only_b"]}


def test_compare_metadata_overlap(store):
    m = store.compare("pap-003", "rep-002")["metadata"]
    assert m["shared_authors"] == ["Tomas Berg"]
    assert set(m["shared_keywords"]) == {"fault detection", "wind turbine", "vibration"}
    assert m["author_jaccard"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["keyword_jaccard"] == pytest.approx(3 / 7, abs=1e-3)
    assert m["same_type"] is False and m["year_gap"] == 0


def test_compare_unknown_document(store):
    with pytest.raises(dr.NotFound):
        store.compare("pap-001", "missing")


# --------------------------------------------------------------------------
# Association analysis
# --------------------------------------------------------------------------
def test_related_finds_topical_neighbours(store):
    rel = store.related("pat-002", limit=3)
    assert "pat-002" not in ids(rel)
    assert set(ids(rel)[:2]) <= {"rep-002", "pap-003", "drw-001"}
    sims = [r["similarity"] for r in rel]
    assert sims == sorted(sims, reverse=True) and all(0 < s <= 1 for s in sims)
    assert ids(store.related("pat-001", limit=1)) == ["pap-004"]


def test_related_agrees_with_compare(store):
    for r in store.related("rep-001", limit=3):
        assert r["similarity"] == pytest.approx(store.compare("rep-001", r["doc_id"])["cosine"], abs=1e-3)


def test_graph_edges_respect_threshold_and_match_compare(store):
    g = store.similarity_graph(0.10)
    assert g["edges"]
    for e in g["edges"]:
        assert e["weight"] >= 0.10
        assert e["weight"] == pytest.approx(store.compare(e["source"], e["target"])["cosine"], abs=1e-3)
    # Every pair at or above the threshold is present, and nothing else.
    expected = {frozenset(p) for p in combinations([d["_id"] for d in store.documents.find()], 2)
                if store.compare(*p)["cosine"] >= 0.10 + 1e-3}
    found = {frozenset((e["source"], e["target"])) for e in g["edges"]}
    assert expected <= found


def test_graph_threshold_is_monotonic_and_groups_topics(store):
    counts = [len(store.similarity_graph(t)["edges"]) for t in (0.0, 0.1, 0.2, 0.5)]
    assert counts == sorted(counts, reverse=True) and counts[-1] == 0
    groups = [set(c) for c in store.similarity_graph(0.10)["components"] if len(c) > 1]
    assert {"pap-003", "pat-002", "rep-002", "drw-001"} in groups
    assert all(not (c & {"pat-001"} and c & {"pap-003"}) for c in groups)


def test_graph_can_be_restricted_by_type(store):
    g = store.similarity_graph(0.0, doc_type="paper")
    assert {n["id"] for n in g["nodes"]} == {"pap-001", "pap-002", "pap-003", "pap-004"}
    assert all(e["source"].startswith("pap") and e["target"].startswith("pap") for e in g["edges"])


def test_keyword_cooccurrence_matches_manual_count(store):
    expected = Counter()
    for d in store.documents.find():
        for a, b in combinations(sorted(d["keywords"]), 2):
            expected[(a, b)] += 1
    got = store.keyword_cooccurrence(limit=1000)
    assert {(r["a"], r["b"]): r["count"] for r in got} == dict(expected)
    assert got[0]["count"] == max(expected.values())
    assert all(r["a"] < r["b"] for r in got)


def test_keyword_cooccurrence_filters(store):
    rows = store.keyword_cooccurrence(min_count=2)
    assert rows and all(r["count"] >= 2 for r in rows)
    wt = store.keyword_cooccurrence(keyword="Wind Turbine")
    assert wt and all("wind turbine" in (r["a"], r["b"]) for r in wt)
    top = next(r for r in wt if {r["a"], r["b"]} == {"wind turbine", "vibration"})
    assert top["docs"] == ["pap-003", "pat-002", "rep-002"]
    only_patents = store.keyword_cooccurrence(doc_type="patent")
    assert all(set(r["docs"]) <= {"pat-001", "pat-002"} for r in only_patents)


# --------------------------------------------------------------------------
# Download round trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("doc_id", ["drw-001", "drw-002", "pap-002", "pat-001"])
def test_download_round_trip_is_byte_identical(store, tmp_path, doc_id):
    original = next(d for d in sample_data.documents() if d["doc_id"] == doc_id)
    r = store.download(doc_id, str(tmp_path / "out"))
    with open(r["path"], "rb") as fh:
        data = fh.read()
    assert data == original["file_bytes"]
    assert hashlib.sha256(data).hexdigest() == r["sha256"] == store.get_document(doc_id)["sha256"]
    assert os.path.basename(r["path"]) == original["filename"]


def test_drawing_is_stored_as_binary(store, tmp_path):
    r = store.download("drw-001", str(tmp_path))
    with open(r["path"], "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    assert store.get_document("drw-001")["content_type"] == "image/png"


def test_download_never_overwrites(store, tmp_path):
    first = store.download("drw-002", str(tmp_path))
    second = store.download("drw-002", str(tmp_path))
    assert first["path"] != second["path"] and os.path.exists(first["path"]) and os.path.exists(second["path"])
    assert "(1)" in os.path.basename(second["path"])


def test_download_from_path_added_file(empty_store, tmp_path):
    src = tmp_path / "part.dxf"
    payload = bytes(range(256)) * 40  # arbitrary non-text bytes, spans several chunks
    src.write_bytes(payload)
    r = empty_store.add_document("Bracket drawing", "drawing", abstract="Bracket outline", file_path=str(src))
    out = empty_store.download(r["doc_id"], str(tmp_path / "dl"))
    assert open(out["path"], "rb").read() == payload


def test_download_without_file_and_unknown_id(empty_store, tmp_path):
    r = empty_store.add_document("No file", "paper", text="text only")
    with pytest.raises(dr.DocError):
        empty_store.download(r["doc_id"], str(tmp_path))
    with pytest.raises(dr.NotFound):
        empty_store.download("ghost", str(tmp_path))


def test_stored_filename_cannot_escape_destination(empty_store, tmp_path):
    r = empty_store.add_document("Sneaky", "paper", text="abc def", file_bytes=b"x", filename="../../evil.txt")
    out = empty_store.download(r["doc_id"], str(tmp_path / "dest"))
    assert os.path.dirname(out["path"]) == str(tmp_path / "dest")


def test_corrupted_gridfs_file_is_detected(store, tmp_path):
    doc = store.get_document("drw-002")
    store.db["fs.chunks"].update_one({"files_id": doc["file_id"], "n": 0}, {"$set": {"data": b"corrupt"}})
    store.db["fs.files"].update_one({"_id": doc["file_id"]}, {"$set": {"length": 7}})
    with pytest.raises(dr.DocError):
        store.download("drw-002", str(tmp_path))
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------
# Delete cleanup
# --------------------------------------------------------------------------
def test_delete_removes_document_postings_and_gridfs_file(store):
    file_id = store.get_document("drw-001")["file_id"]
    n_postings = store.postings.count_documents({"doc_id": "drw-001"})
    assert n_postings > 0
    r = store.delete_document("drw-001")
    assert r == {"doc_id": "drw-001", "postings_removed": n_postings, "file_removed": True}
    assert store.documents.count_documents({"_id": "drw-001"}) == 0
    assert store.postings.count_documents({"doc_id": "drw-001"}) == 0
    assert store.db["fs.files"].count_documents({"_id": file_id}) == 0
    assert store.db["fs.chunks"].count_documents({"files_id": file_id}) == 0
    with pytest.raises(dr.NotFound):
        store.get_document("drw-001")


def test_deleted_document_disappears_from_search_and_analysis(store):
    assert "pat-001" in ids(store.search("posting list", snippet=False))
    store.delete_document("pat-001")
    assert "pat-001" not in ids(store.search("posting list", snippet=False))
    assert all("pat-001" not in r["docs"] for r in store.keyword_cooccurrence())
    assert all("pat-001" not in (e["source"], e["target"]) for e in store.similarity_graph(0.0)["edges"])
    assert "pat-001" not in ids(store.related("pap-004", limit=9))
    with pytest.raises(dr.NotFound):
        store.compare("pat-001", "pap-004")


def test_delete_leaves_other_documents_untouched_and_updates_counters(store):
    before_scores = {r["doc_id"]: r["score"] for r in store.search("vibration", snippet=False) if r["doc_id"] != "pat-002"}
    n, total = store._corpus()
    length = store.get_document("pat-002")["length"]
    store.delete_document("pat-002")
    assert store._corpus() == (n - 1, total - length)
    assert store.documents.count_documents({}) == 9
    assert store.db["fs.files"].count_documents({}) == 9
    # Every posting left still belongs to a stored document, and every file to a document.
    live = {d["_id"] for d in store.documents.find({}, {"_id": 1})}
    assert set(store.postings.distinct("doc_id")) <= live
    assert {f["metadata"]["doc_id"] for f in store.db["fs.files"].find()} == live
    assert set(before_scores) <= set(ids(store.search("vibration", snippet=False)))


def test_delete_everything_leaves_no_residue(store):
    for d in [x["_id"] for x in store.documents.find({}, {"_id": 1})]:
        store.delete_document(d)
    assert store.documents.count_documents({}) == 0
    assert store.postings.count_documents({}) == 0
    assert store.db["fs.files"].count_documents({}) == 0
    assert store.db["fs.chunks"].count_documents({}) == 0
    assert store._corpus() == (0, 0)
    assert store.search("vibration") == []


def test_delete_unknown_id_and_repeat(store):
    with pytest.raises(dr.NotFound):
        store.delete_document("ghost")
    store.delete_document("rep-001")
    with pytest.raises(dr.NotFound):
        store.delete_document("rep-001")


def test_delete_can_be_resumed_after_partial_failure(store):
    """Simulate a crash after the GridFS file was removed but before the record was."""
    doc = store.get_document("rep-002")
    store.postings.delete_many({"doc_id": "rep-002"})
    store.fs.delete(doc["file_id"])
    r = store.delete_document("rep-002")
    assert r["postings_removed"] == 0 and r["file_removed"] is False
    assert store.documents.count_documents({"_id": "rep-002"}) == 0


def test_rebuild_corpus_stats_repairs_counters(store):
    store.meta.update_one({"_id": "corpus"}, {"$set": {"n": 999, "total_len": 1}})
    store.rebuild_corpus_stats()
    assert store._corpus() == (10, sum(d["length"] for d in store.documents.find()))


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def test_stats(store):
    s = store.stats()
    assert s["documents"] == 10 and s["by_type"] == {"drawing": 2, "paper": 4, "patent": 2, "report": 2}
    assert s["gridfs_files"] == 10 and s["gridfs_bytes"] == sum(len(d["file_bytes"]) for d in sample_data.documents())
    assert s["postings"] == store.postings.count_documents({})
    assert s["distinct_terms"] == len(store.postings.distinct("term"))
    assert s["year_range"] == (2020, 2024) and s["distinct_authors"] == 6
    assert dict(s["top_keywords"])["wind turbine"] == 4


def test_stats_on_empty_database(empty_store):
    s = empty_store.stats()
    assert s["documents"] == 0 and s["postings"] == 0 and s["by_type"] == {} and s["year_range"] is None
