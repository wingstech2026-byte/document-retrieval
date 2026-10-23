"""Tests for the advanced features: query language, facets, paging, spelling suggestions,
score explanation, document insights, analytics, citations and duplicate detection.
They need a real MongoDB, like test_docretrieval.py (whose fixtures they reuse)."""
from collections import Counter
from itertools import combinations

import pytest

import docretrieval as dr
import sample_data
from test_docretrieval import empty_store, ids, mongo_client, store  # noqa: F401  (fixtures)


# --------------------------------------------------------------------------
# Query language
# --------------------------------------------------------------------------
def test_parse_query_understands_every_operator():
    pq = dr.parse_query('"wind turbine" -sensor type:Technical-Report author:"tomas berg" keyword:gearbox '
                        'title:mount year:2020..2023 fault http://x.org')
    assert pq.phrases == ["wind turbine"] and pq.exclude == ["sensor"]
    assert pq.doc_type == "report" and pq.authors == ["tomas berg"] and pq.keywords == ["gearbox"]
    assert pq.title_terms == ["mount"] and (pq.year_from, pq.year_to) == (2020, 2023)
    assert {"wind", "turbine", "fault"} <= set(pq.terms) and "sensor" not in pq.terms
    assert "http" in pq.terms  # an unknown "field:" is just text


@pytest.mark.parametrize("expr, expected", [("2022", (2022, 2022)), ("2020..2023", (2020, 2023)), ("2021..", (2021, None)),
                                            ("..2023", (None, 2023)), (">2020", (2021, None)), (">=2020", (2020, None)),
                                            ("<2023", (None, 2022)), ("<=2023", (None, 2023))])
def test_year_filter_forms(expr, expected):
    pq = dr.parse_query(f"year:{expr}")
    assert (pq.year_from, pq.year_to) == expected


def test_malformed_operators_are_reported():
    for bad in ("year:soon", "type:novel", "-type:patent"):
        with pytest.raises(dr.DocError):
            dr.parse_query(bad)


def test_phrase_search_needs_the_exact_words_in_order(store):  # noqa: F811
    assert ids(store.search('"posting list"', snippet=False)) == ["pat-001"]
    assert store.search('"list posting"', snippet=False) == []
    assert len(store.search("wind turbine", snippet=False)) > 1
    assert ids(store.search('"wind turbine" detectors', snippet=False))[0] == "pap-003"


def test_exclusion_removes_documents_containing_the_word(store):  # noqa: F811
    base = set(ids(store.search("vibration", snippet=False)))
    kept = set(ids(store.search("vibration -federated", snippet=False)))
    assert "pap-003" in base and kept == base - {"pap-003"}
    assert store.search("vibration -sensor", snippet=False) == []  # every vibration document mentions sensors


def test_field_filters(store):  # noqa: F811
    assert ids(store.search("type:patent vibration", snippet=False)) == ["pat-002"]
    assert set(ids(store.search("author:berg year:>=2023", snippet=False))) == {"pap-003", "rep-002", "drw-001"}
    assert set(ids(store.search('keyword:"wind turbine"', snippet=False))) == {"pap-003", "pat-002", "rep-002", "drw-001"}
    assert set(ids(store.search("title:gearbox", snippet=False))) == {"pap-003", "rep-002", "drw-001"}
    assert ids(store.search("title:gearbox year:2022", snippet=False)) == []
    # a keyword argument and a query filter are ANDed
    assert ids(store.search("type:patent vibration", doc_type="report", snippet=False)) == []
    assert ids(store.search("type:patent vibration", doc_type="patent", snippet=False)) == ["pat-002"]


def test_filters_alone_list_matching_documents(store):  # noqa: F811
    r = store.search_ex("type:patent")
    assert r["total"] == 2 and {x["doc_id"] for x in r["results"]} == {"pat-001", "pat-002"}
    assert all(x["score"] == 0 for x in r["results"])
    assert store.search_ex("year:1800..1801")["total"] == 0


def test_facets_count_the_whole_result_set(store):  # noqa: F811
    r = store.search_ex("vibration", limit=2)
    allr = store.search_ex("vibration", limit=100)
    assert r["total"] == allr["total"] == len(allr["results"]) and len(r["results"]) == 2
    f = r["facets"]
    assert sum(n for _, n in f["type"]) == r["total"] and dict(f["type"])["patent"] == 1
    assert dict(f["author"])["Tomas Berg"] == len([x for x in allr["results"] if "Tomas Berg" in x["authors"]])
    assert f["year"] == sorted(f["year"], reverse=True)


def test_paging_and_sorting(store):  # noqa: F811
    full = ids(store.search_ex("document", limit=100)["results"])
    pages = [ids(store.search_ex("document", limit=2, offset=o)["results"]) for o in (0, 2, 4)]
    assert sum(pages, []) == full[:6] and all(len(p) == 2 for p in pages[:2])
    by_year = store.search_ex("vibration", sort="year", limit=100)["results"]
    years = [r["year"] for r in by_year]
    assert years == sorted(years, reverse=True) and len(set(ids(by_year))) == len(by_year)


def test_spelling_suggestions(store):  # noqa: F811
    assert store.suggest("gearbxo vibraton") == "gearbox vibration"
    assert store.search_ex("gearbxo vibraton")["suggestion"] == "gearbox vibration"
    assert store.suggest("gearbox vibration") is None
    assert store.suggest("xqzvwkpt") is None
    assert store.suggest('"gearbxo" -vibraton type:patent') is None  # phrases, exclusions and filters are left alone
    assert store.search_ex("posting listt")["suggestion"] == "posting list"


def test_suggestion_vocabulary_follows_changes(empty_store):  # noqa: F811
    s = empty_store
    assert s.suggest("zirconium") is None
    r = s.add_document("Crucible", "report", text="zirconium crucible thermal shock")
    assert s.suggest("zirconim") == "zirconium"
    s.delete_document(r["doc_id"])
    assert s.suggest("zirconim") is None


# --------------------------------------------------------------------------
# Explain, insights, analytics, citations, duplicates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("query", ["vibration sensor", "inverted index compression", '"posting list" -federated year:>2019'])
def test_explain_adds_up_to_the_search_score(store, query):  # noqa: F811
    for hit in store.search(query, snippet=False)[:3]:
        e = store.explain(query, hit["doc_id"])
        assert e["score"] == pytest.approx(hit["score"], abs=1e-3)
        assert e["score"] == pytest.approx(sum(r["score"] for r in e["rows"]), abs=1e-3)


def test_explain_details(store):  # noqa: F811
    e = store.explain("vibration federated", "pat-002")
    rows = {r["term"]: r for r in e["rows"]}
    assert rows["vibration"]["in_title"] and rows["vibration"]["in_keywords"]
    assert rows["federated"]["tf"] == 0 and rows["federated"]["score"] == 0
    assert rows["vibration"]["df"] == store.postings.count_documents({"term": "vibration"})
    assert e["corpus_size"] == 10 and e["k1"] == dr.K1 and e["b"] == dr.B


def test_insights(store):  # noqa: F811
    i = store.insights("pap-003")
    text = " ".join(store.get_document("pap-003")["text"].split())
    assert 1 <= len(i["summary"]) <= 3 and all(s in text for s in i["summary"])
    assert i["summary"] == sorted(i["summary"], key=text.index)  # kept in reading order
    assert "farm" in [t for t, _ in i["key_terms"]] and len(i["key_terms"]) == 10
    assert i["words"] == len(text.split()) and i["reading_minutes"] == 1
    assert i["distinct_terms"] == store.postings.count_documents({"doc_id": "pap-003"})
    listed = {t for k in store.get_document("pap-003")["keywords"] for t in dr.tokenize(k)}
    assert not set(i["suggested_keywords"]) & listed


def test_insights_for_tiny_text(empty_store):  # noqa: F811
    r = empty_store.add_document("Tiny", "drawing", text="Just a short note.")
    i = empty_store.insights(r["doc_id"])
    assert i["summary"] == [] and i["reading_minutes"] == 1


def test_analytics_matches_manual_counts(store):  # noqa: F811
    an = store.analytics()
    docs = list(store.documents.find())
    manual = Counter(d["year"] for d in docs)
    assert {r["year"]: r["total"] for r in an["by_year"]} == dict(manual)
    assert [r["year"] for r in an["by_year"]] == sorted(manual)
    row = next(r for r in an["by_year"] if r["year"] == 2023)
    assert row["drawing"] == 2 and row["paper"] == 1 and row["report"] == 1 and row["patent"] == 0
    authors = Counter(a for d in docs for a in d["authors"])
    assert dict(an["top_authors"]) == dict(authors) and an["top_authors"][0] == ("Tomas Berg", 4)
    pairs = Counter(p for d in docs for p in combinations(sorted(d["authors"]), 2))
    assert {(e["a"], e["b"]): e["count"] for e in an["author_edges"]} == dict(pairs)
    counts = [e["count"] for e in an["author_edges"]]
    assert counts == sorted(counts, reverse=True)


def test_author_network_filter(store):  # noqa: F811
    rows = store.author_network(author="Tomas Berg")
    assert rows and all("Tomas Berg" in (r["a"], r["b"]) for r in rows)
    assert store.author_network(min_count=2) == []


def test_citations(store, empty_store):  # noqa: F811
    bib = store.citation("pap-003", "bibtex")
    assert bib.startswith("@article{pap_003,") and "author = {Priya Nair and Tomas Berg}" in bib and "year = {2024}" in bib
    assert store.citation("rep-001", "bibtex").startswith("@techreport{")
    assert "howpublished = {Patent}" in store.citation("pat-001", "bibtex")
    ris = store.citation("rep-001", "ris")
    assert ris.startswith("TY  - RPRT\r\n") and ris.count("AU  - ") == 2 and ris.rstrip().endswith("ER  -")
    assert store.citation("pat-001", "apa") == ("Alvarez, S. (2020). Method for Compressing Posting Lists Using Block-wise "
                                                "Delta Coding [Patent]. pat-001.\n")
    assert store.citation("pap-003", "apa").startswith("Nair, P., & Berg, T. (2024).")
    with pytest.raises(dr.DocError):
        store.citation("pap-003", "endnote")
    r = empty_store.add_document("R&D costs: 50% of A_B", "paper", authors=["Ann Lee"], text="x y z")
    assert r"R\&D costs: 50\% of A\_B" in empty_store.citation(r["doc_id"], "bibtex")


def test_duplicate_detection(store):  # noqa: F811
    original = next(d for d in sample_data.documents() if d["doc_id"] == "pat-001")
    store.add_document(**{**original, "doc_id": "pat-copy", "title": original["title"] + " (revised)"})
    pairs = store.find_duplicates(0.9)
    assert [{p["a"]["id"], p["b"]["id"]} for p in pairs] == [{"pat-001", "pat-copy"}]
    assert pairs[0]["similarity"] > 0.95
    assert store.related("pat-copy", 1)[0]["doc_id"] == "pat-001"
    strict = {frozenset((p["a"]["id"], p["b"]["id"])) for p in store.find_duplicates(0.9)}
    loose = {frozenset((p["a"]["id"], p["b"]["id"])) for p in store.find_duplicates(0.2)}
    assert strict <= loose and len(loose) > len(strict)
