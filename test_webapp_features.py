"""Web tests for the advanced features: facets, paging, suggestions, explain page, insights,
exports, citations, analytics, duplicates and the JSON API (real MongoDB, throw-away database)."""
import csv
import io
import json

import pytest

import docretrieval as dr
import sample_data
from test_docretrieval import empty_store, mongo_client, store  # noqa: F401  (fixtures)
from webapp import create_app


def make_client(st, read_only=False):
    app = create_app(st, read_only=read_only)
    app.testing = True
    return app.test_client()


@pytest.fixture
def client(store):  # noqa: F811
    return make_client(store)


def text(resp):
    return resp.get_data(as_text=True)


# ---- search page ---------------------------------------------------------------
def test_query_syntax_works_in_the_search_box(client):
    html = text(client.get('/?q="posting list"'))
    assert "pat-001" in html and "pap-004" not in html and "“posting list”" in html
    html = text(client.get("/?q=vibration -federated type:report"))
    assert "rep-002" in html and "pap-003" not in html and "not federated" in html and "type: report" in html


def test_facets_are_links_that_refine_the_search(client):
    html = text(client.get("/?q=vibration"))
    assert "Refine results" in html and "q=vibration+type:patent" in html and "author:%22Tomas+Berg%22" in html
    refined = text(client.get("/?q=vibration+type:patent"))
    assert "pat-002" in refined and "rep-002" not in refined
    kept = text(client.get("/?q=vibration&sort=year"))
    assert "q=vibration+type:patent&amp;sort=year" in kept  # the facet links keep the other settings


def test_did_you_mean(client):
    html = text(client.get("/?q=gearbxo+vibraton"))
    assert "Did you mean" in html and "gearbox vibration" in html and "No documents match" in html
    followed = text(client.get("/?q=gearbox+vibration"))
    assert "Did you mean" not in followed and "drw-001" in followed


def test_results_link_to_the_score_explanation(client):
    html = text(client.get("/?q=vibration+sensor"))
    assert "why this score?" in html and "/explain/pat-002?q=vibration+sensor" in html
    page = text(client.get("/explain/pat-002?q=vibration+sensor"))
    assert "3.286" in page and "vibration" in page and "sensor" in page and "title" in page and "k1 = 1.5" in page
    assert client.get("/explain/pat-002").status_code == 400
    assert client.get("/explain/ghost?q=vibration").status_code == 404


def test_pagination(empty_store):  # noqa: F811
    for i in range(25):
        empty_store.add_document(f"Report number {i}", "report", year=2000 + i, text="calibration notes " * (i + 1), doc_id=f"r{i:02d}")
    c = make_client(empty_store)
    p1, p3 = text(c.get("/?q=calibration")), text(c.get("/?q=calibration&page=3"))
    assert "25 results" in p1 and "Page 1 of 3" in p1 and p1.count('class="result"') == 10
    assert "Next" in p1 and "Previous" not in p1
    assert "Page 3 of 3" in p3 and p3.count('class="result"') == 5 and "Next" not in p3 and "Previous" in p3
    newest = text(c.get("/?q=calibration&sort=year"))
    assert newest.index("r24") < newest.index("r23")


def test_search_page_survives_odd_input(client):
    for q in ('"', "-", "type:", "year:..", "author:", ":::", "title:", "<script>"):
        assert client.get("/", query_string={"q": q}).status_code in (200, 400), q
    assert "<script>" not in text(client.get("/", query_string={"q": "<script>alert(1)</script>"}))
    assert client.get("/?q=year:soon").status_code == 400


# ---- exports and citations ---------------------------------------------------------
def test_csv_export(client):
    r = client.get("/export?q=vibration&fmt=csv")
    assert r.mimetype == "text/csv" and "attachment" in r.headers["Content-Disposition"]
    rows = list(csv.reader(io.StringIO(text(r).lstrip("﻿"))))
    assert rows[0] == ["id", "type", "year", "title", "authors", "score"]
    assert {row[0] for row in rows[1:]} == {"pap-003", "pat-002", "rep-002", "drw-001"}
    assert float(rows[1][5]) >= float(rows[-1][5])


def test_csv_export_is_safe_against_formula_injection(empty_store):  # noqa: F811
    empty_store.add_document("=HYPERLINK(\"http://x\")", "paper", authors=["@evil"], text="payload payload", doc_id="p1")
    rows = list(csv.reader(io.StringIO(text(make_client(empty_store).get("/export?q=payload")).lstrip("﻿"))))
    assert rows[1][3].startswith("'=") and rows[1][4].startswith("'@")


def test_json_export_respects_filters(client):
    data = json.loads(text(client.get("/export?q=vibration+type:patent&fmt=json")))
    assert data["total"] == 1 and data["results"][0]["doc_id"] == "pat-002"


@pytest.mark.parametrize("fmt, ext, start", [("bibtex", "bib", "@article{pap_003,"), ("ris", "ris", "TY  - JOUR"), ("apa", "txt", "Nair, P.")])
def test_citation_download(client, fmt, ext, start):
    r = client.get(f"/doc/pap-003/cite/{fmt}")
    assert r.status_code == 200 and text(r).startswith(start)
    assert f'filename="pap-003.{ext}"' in r.headers["Content-Disposition"] and "utf-8" in r.headers["Content-Type"]
    assert client.get("/doc/pap-003/cite/endnote").status_code == 400
    assert client.get("/doc/ghost/cite/bibtex").status_code == 404


# ---- document page, analytics, duplicates --------------------------------------------
def test_document_page_has_insights_and_citation_links(client):
    html = text(client.get("/doc/pap-003"))
    assert "At a glance" in html and "Key terms" in html and "min read" in html and "Summary" in html
    assert "farm" in html and "/doc/pap-003/cite/bibtex" in html and "/api/doc/pap-003" in html


def test_analytics_page(client):
    html = text(client.get("/analytics"))
    assert "Documents per year" in html and "<svg" in html and "Co-authorship network" in html
    assert "Tomas Berg" in html and "Show as a table" in html
    assert html.count("<rect") >= 5 and "aria-label" in html


def test_duplicates_page(client, store):  # noqa: F811
    assert "No pairs reach" in text(client.get("/duplicates?threshold=0.9"))
    original = next(d for d in sample_data.documents() if d["doc_id"] == "pat-001")
    store.add_document(**{**original, "doc_id": "pat-copy"})
    html = text(client.get("/duplicates?threshold=0.9"))
    assert "pat-001" in html and "pat-copy" in html
    assert client.get("/duplicates?threshold=x").status_code == 400


def test_adding_a_near_copy_warns(empty_store):  # noqa: F811
    c = make_client(empty_store)
    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 8
    first = c.post("/add", data={"title": "Original study", "type": "paper", "text": body, "keywords": "x"})
    assert "Warning" not in first.headers["Location"]
    second = c.post("/add", data={"title": "Original study again", "type": "paper", "text": body, "keywords": "x"})
    assert second.status_code == 302 and "Warning" in second.headers["Location"] and "duplicate" in second.headers["Location"]
    unrelated = c.post("/add", data={"title": "Soil chemistry", "type": "report", "text": "nitrogen phosphorus soil sample " * 8})
    assert "Warning" not in unrelated.headers["Location"]


# ---- JSON API ------------------------------------------------------------------------
def test_api_search(client):
    r = client.get("/api/search?q=vibration -federated&limit=2")
    assert r.status_code == 200 and r.mimetype == "application/json"
    d = r.get_json()
    assert d["total"] == 3 and len(d["results"]) == 2 and d["interpreted"]["exclude"] == ["federated"]
    assert {"doc_id", "title", "type", "year", "authors", "score", "snippet"} <= set(d["results"][0])
    assert {f["value"] for f in d["facets"]["type"]} == {"patent", "report", "drawing"}
    assert client.get("/api/search?q=gearbxo").get_json()["suggestion"] == "gearbox"
    assert len(client.get("/api/search?q=document&limit=1000").get_json()["results"]) <= 100
    page2 = client.get("/api/search?q=vibration&limit=2&offset=2").get_json()
    assert len(page2["results"]) == 2 and page2["total"] == 4


def test_api_documents(client):
    d = client.get("/api/doc/drw-001").get_json()
    assert d["id"] == "drw-001" and d["has_file"] and d["content_type"] == "image/png" and d["download"].endswith("/doc/drw-001/file")
    assert "text" not in d and "_id" not in d and "file_id" not in d
    assert "bearing bores" in client.get("/api/doc/drw-001?text=1").get_json()["text"]
    assert client.get("/api/related/pat-002?limit=2").get_json()["related"][0]["doc_id"] == "rep-002"
    assert client.get("/api/compare?a=pap-003&b=rep-002").get_json()["cosine"] == pytest.approx(0.2262, abs=1e-3)
    assert len(client.get("/api/insights/pap-003").get_json()["key_terms"]) == 10
    assert client.get("/api/explain/pat-002?q=vibration+sensor").get_json()["score"] == pytest.approx(3.2859, abs=1e-3)
    assert client.get("/api/stats").get_json()["documents"] == 10


def test_api_errors_are_json(client):
    for url, code in (("/api/doc/ghost", 404), ("/api/compare?a=pap-003&b=ghost", 404), ("/api/search?q=year:soon", 400)):
        r = client.get(url)
        assert r.status_code == code and r.mimetype == "application/json" and "error" in r.get_json()


def test_api_docs_page(client):
    html = text(client.get("/api"))
    assert "/api/search" in html and "curl" in html


def test_read_only_site_keeps_every_new_read_feature(store):  # noqa: F811
    c = make_client(store, read_only=True)
    for url in ("/analytics", "/duplicates", "/explain/pat-002?q=vibration", "/export?q=vibration", "/api/search?q=vibration",
                "/doc/pap-003/cite/bibtex", "/api", "/?q=vibration+type:patent"):
        assert c.get(url).status_code == 200, url
    assert c.post("/doc/pap-003/delete").status_code == 403
