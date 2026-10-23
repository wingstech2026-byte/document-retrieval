"""Tests for the Flask front end (real MongoDB, throw-away database)."""
import io

import pytest

import docretrieval as dr
import sample_data
from test_docretrieval import empty_store, mongo_client, store  # noqa: F401  (fixtures)
from webapp import create_app


@pytest.fixture
def client(store):  # noqa: F811
    app = create_app(store)
    app.testing = True
    return app.test_client()


@pytest.fixture
def empty_client(empty_store):  # noqa: F811
    app = create_app(empty_store)
    app.testing = True
    return app.test_client()


def text(resp):
    return resp.get_data(as_text=True)


def test_every_page_renders(client):
    for url in ("/", "/add", "/compare", "/graph", "/keywords", "/stats", "/doc/pap-001"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert "<title>" in text(r)
    assert client.get("/compare?a=pap-003&b=rep-002").status_code == 200


def test_home_lists_all_documents_without_query(client):
    html = text(client.get("/"))
    for d in sample_data.documents():
        assert d["title"].replace("&", "&amp;") in html or d["title"] in html
    only = text(client.get("/?type=patent"))
    assert "pat-001" in only and "pap-001" not in only


def test_search_shows_ranked_results_with_highlighted_snippets(client):
    html = text(client.get("/?q=inverted+index+compression"))
    assert html.index("pat-001") < html.index("pap-004")
    assert "<mark>inverted</mark>" in html and "ranked by BM25" in html


def test_search_filters(client):
    html = text(client.get("/?q=vibration&type=patent"))
    assert "pat-002" in html and "rep-002" not in html
    html = text(client.get("/?q=vibration&year_to=2022"))
    assert "pat-002" in html and "pap-003" not in html
    html = text(client.get("/?q=sensor+vibration&author=berg"))
    assert "pat-002" in html and "pap-001" not in html
    assert "No documents match" in text(client.get("/?q=xyzzyplugh"))


def test_bad_year_is_reported_not_crashed(client):
    r = client.get("/?q=index&year_from=abc")
    assert r.status_code == 400 and "whole number" in text(r)


def test_document_page_reading_and_paging(client):
    long = client.get("/doc/pap-001")
    assert "Page 1 of" in text(long) and "Next" in text(long)
    second = text(client.get("/doc/pap-001?page=2"))
    assert "Page 2 of" in second and "Previous" in second
    assert client.get("/doc/pap-001?page=999").status_code == 400
    assert client.get("/doc/nope").status_code == 404


def test_document_page_shows_metadata_related_and_drawing(client):
    html = text(client.get("/doc/pat-002"))
    assert "Tomas Berg" in html and "wind turbine" in html and "Related documents" in html and "rep-002" in html
    d = text(client.get("/doc/drw-001"))
    assert "<img" in d and "/doc/drw-001/file?inline=1" in d


def test_file_download_is_byte_identical(client):
    original = next(d for d in sample_data.documents() if d["doc_id"] == "drw-001")["file_bytes"]
    r = client.get("/doc/drw-001/file")
    assert r.data == original
    assert r.headers["Content-Disposition"].startswith("attachment") and "gearbox_housing_sheet3.png" in r.headers["Content-Disposition"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    inline = client.get("/doc/drw-001/file?inline=1")
    assert inline.headers["Content-Disposition"].startswith("inline") and inline.mimetype == "image/png"
    assert client.get("/doc/ghost/file").status_code == 404


def test_active_content_is_never_served_inline(empty_client, empty_store):  # noqa: F811
    empty_store.add_document("Page", "report", text="hello world", file_bytes=b"<script>alert(1)</script>",
                             filename="x.html", doc_id="rep-html")
    r = empty_client.get("/doc/rep-html/file?inline=1")
    assert r.headers["Content-Disposition"].startswith("attachment")


def test_text_is_escaped_everywhere(empty_client, empty_store):  # noqa: F811
    empty_store.add_document("<script>alert(1)</script> Widget", "paper", authors=["<b>Eve</b>"],
                             text="a widget <img src=x onerror=alert(2)> widget", doc_id="pap-xss")
    for url in ("/", "/?q=widget", "/doc/pap-xss"):
        html = text(empty_client.get(url))
        assert "<script>alert(1)" not in html and "<img src=x" not in html and "<b>Eve" not in html, url
    assert "&lt;script&gt;" in text(empty_client.get("/?q=widget"))


def test_compare_page(client):
    html = text(client.get("/compare?a=pap-003&b=rep-002"))
    assert "0.226" in html and "Tomas Berg" in html and "federated" in html and "baseline" in html
    assert client.get("/compare?a=pap-003&b=ghost").status_code == 404


def test_graph_page(client):
    html = text(client.get("/graph?threshold=0.10"))
    assert "11 links" in html and html.count("<line") == 11 and "<svg" in html
    assert "Group 1" in html
    assert "0 links" in text(client.get("/graph?threshold=0.9"))
    assert client.get("/graph?threshold=zzz").status_code == 400


def test_keywords_page(client):
    html = text(client.get("/keywords?keyword=wind+turbine&min=2"))
    assert "gearbox" in html and "vibration" in html and "information retrieval" not in html


def test_stats_page(client):
    html = text(client.get("/stats"))
    assert ">10<" in html and "11,733" not in html and "GridFS" in html


def test_add_with_file_upload_then_search_download_delete(empty_client, empty_store, tmp_path):  # noqa: F811
    payload = bytes(range(256)) * 20
    r = empty_client.post("/add", data={
        "title": "Bracket drawing rev B", "type": "drawing", "year": "2024", "authors": "A. Rivera; B. Chen",
        "keywords": "bracket, steel", "abstract": "Steel mounting bracket, two bolt holes.", "text": "",
        "file": (io.BytesIO(payload), "bracket.dxf")}, content_type="multipart/form-data")
    assert r.status_code == 302
    doc_id = r.headers["Location"].split("/doc/")[1].split("?")[0]

    assert doc_id in text(empty_client.get("/?q=bracket"))
    assert empty_client.get(f"/doc/{doc_id}/file").data == payload
    assert empty_store.get_document(doc_id)["authors"] == ["A. Rivera", "B. Chen"]

    r = empty_client.post(f"/doc/{doc_id}/delete")
    assert r.status_code == 302
    assert empty_store.postings.count_documents({"doc_id": doc_id}) == 0
    assert empty_store.db["fs.files"].count_documents({}) == 0
    assert empty_client.get(f"/doc/{doc_id}").status_code == 404


def test_add_validation_error_keeps_form_values(empty_client, empty_store):  # noqa: F811
    r = empty_client.post("/add", data={"title": "", "type": "paper", "keywords": "kept, values"})
    assert r.status_code == 400 and "title is required" in text(r) and "kept, values" in text(r)
    r = empty_client.post("/add", data={"title": "x", "type": "paper", "year": "twenty"})
    assert r.status_code == 400
    assert empty_store.documents.count_documents({}) == 0


def test_delete_requires_post(client):
    assert client.get("/doc/pap-001/delete").status_code == 405
    assert client.post("/doc/ghost/delete").status_code == 404


def test_load_samples_button_is_repeatable(empty_client, empty_store):  # noqa: F811
    assert "Load the 10 sample documents" in text(empty_client.get("/"))
    for _ in range(2):
        assert empty_client.post("/load-samples").status_code == 302
    assert empty_store.documents.count_documents({}) == 10
    assert empty_store.db["fs.files"].count_documents({}) == 10
    assert "Loaded 10 sample documents." in text(empty_client.get("/?msg=Loaded+10+sample+documents."))


def test_empty_database_pages_do_not_crash(empty_client):
    for url in ("/", "/graph", "/keywords", "/stats", "/compare"):
        assert empty_client.get(url).status_code == 200, url
