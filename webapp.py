#!/usr/bin/env python3
"""Web front end for the document retrieval system (Flask).

    python webapp.py                       # http://127.0.0.1:5000, database 'docretrieval'
    python webapp.py --db docretrieval_demo

The page code only calls DocumentStore from docretrieval.py; all data stays in MongoDB/GridFS.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from urllib.parse import quote

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from markupsafe import Markup, escape

import docretrieval as dr

PAGE_CHARS = 1000
PER_PAGE = 10
MAX_GRAPH_NODES = 80
DUP_THRESHOLD = 0.6          # a newly added document this similar to an existing one triggers a warning
# Served inline in the browser; everything else is forced to download so an uploaded
# HTML/SVG file can never run script on this site's origin.
INLINE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain"}
CITE_FORMATS = {"bibtex": ("bib", "application/x-bibtex"), "ris": ("ris", "application/x-research-info-systems"),
                "apa": ("txt", "text/plain")}


def _int(name: str) -> int | None:
    raw = (request.args.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise dr.DocError(f"{name.replace('_', ' ')} must be a whole number")


def _float(name: str, default: float) -> float:
    raw = (request.args.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        raise dr.DocError(f"{name} must be a number")


def _csv_safe(value) -> str:
    """Stop spreadsheet programs from running a cell as a formula."""
    v = "" if value is None else str(value)
    return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v


def _circle_layout(names, cx=360, cy=330, r=250):
    pos = {}
    for i, name in enumerate(names):
        ang = 2 * math.pi * i / max(len(names), 1) - math.pi / 2
        c = math.cos(ang)
        pos[name] = {"x": cx + r * c, "y": cy + r * math.sin(ang), "lx": cx + (r + 30) * c,
                     "ly": cy + (r + 30) * math.sin(ang) + 4,
                     "anchor": "middle" if abs(c) < 0.2 else ("start" if c > 0 else "end")}
    return pos


def create_app(store: dr.DocumentStore, read_only: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
    app.config["STORE"] = store
    app.config["READ_ONLY"] = read_only

    @app.template_filter("hl")
    def highlight(text: str) -> Markup:
        """Snippet markers (**word**) -> <mark>, after escaping everything else."""
        return Markup(re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", str(escape(text or "")), flags=re.S))

    @app.template_filter("size")
    def human_size(n) -> str:
        n = float(n or 0)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        return resp

    @app.before_request
    def block_writes():
        """Read-only mode (public deployments): no uploads, deletes or sample loading."""
        if read_only and (request.method not in ("GET", "HEAD") or request.endpoint == "add"):
            return render_template("error.html", message="This site is read-only: documents can be searched, "
                                   "read, compared and downloaded, but not added or deleted.", code=403), 403

    @app.errorhandler(dr.DocError)
    def doc_error(e):
        code = 404 if isinstance(e, dr.NotFound) else 400
        if request.path.startswith("/api/"):
            return jsonify({"error": str(e)}), code
        return render_template("error.html", message=str(e), code=code), code

    @app.errorhandler(413)
    def too_big(_):
        return render_template("error.html", message="That file is larger than the 64 MB upload limit.", code=413), 413

    @app.context_processor
    def common():
        return {"db_name": store.db.name, "doc_types": dr.DOC_TYPES, "msg": request.args.get("msg"), "read_only": read_only}

    # ---- search / browse ----------------------------------------------------------
    def search_args() -> dict:
        """The search parameters shared by the page, the export and the API."""
        return {"query": (request.args.get("q") or "").strip(), "doc_type": request.args.get("type") or None,
                "year_from": _int("year_from"), "year_to": _int("year_to"),
                "author": (request.args.get("author") or "").strip() or None,
                "require_all": bool(request.args.get("all")),
                "sort": "year" if request.args.get("sort") == "year" else "relevance"}

    @app.get("/")
    def home():
        a = search_args()
        form = {"q": a["query"], "type": a["doc_type"] or "", "year_from": request.args.get("year_from", ""),
                "year_to": request.args.get("year_to", ""), "author": a["author"] or "", "all": a["require_all"],
                "sort": a["sort"]}
        active = bool(a["query"] or a["doc_type"] or a["year_from"] is not None or a["year_to"] is not None or a["author"])
        page = max(_int("page") or 1, 1)
        res = docs = None
        facets = []
        if active:
            res = store.search_ex(limit=PER_PAGE, offset=(page - 1) * PER_PAGE, **a)

            keep = {k: v for k, v in request.args.items() if k not in ("page", "msg")}

            def add_filter(token: str) -> str:
                return url_for("home", **{**keep, "q": (a["query"] + " " + token).strip()})

            def q(v: str) -> str:
                return f'"{v}"' if " " in v else v
            if res["suggestion"]:
                res["suggestion_href"] = url_for("home", **{**keep, "q": res["suggestion"]})
            raw = res["facets"]
            facets = [
                {"title": "Type", "items": [(t, n, add_filter(f"type:{t}")) for t, n in raw["type"]]},
                {"title": "Year", "items": [(y, n, add_filter(f"year:{y}")) for y, n in raw["year"]]},
                {"title": "Author", "items": [(x, n, add_filter(f"author:{q(x)}")) for x, n in raw["author"]]},
                {"title": "Keyword", "items": [(x, n, add_filter(f"keyword:{q(x)}")) for x, n in raw["keyword"]]},
            ]
        else:
            docs = store.list_documents(limit=500)
        base = {k: v for k, v in request.args.items() if k not in ("page", "msg")}
        return render_template("index.html", form=form, res=res, docs=docs, facets=facets, page=page,
                               pages=max(1, math.ceil(res["total"] / PER_PAGE)) if res else 1, base=base, per_page=PER_PAGE)

    @app.get("/explain/<doc_id>")
    def explain(doc_id):
        q = (request.args.get("q") or "").strip()
        if not q:
            raise dr.DocError("give a query to explain, e.g. /explain/pap-001?q=inverted+index")
        e = store.explain(q, doc_id)
        top = max((r["score"] for r in e["rows"]), default=0) or 1
        return render_template("explain.html", e=e, q=q, top=top)

    @app.get("/export")
    def export():
        a = search_args()
        res = store.search_ex(limit=1000, snippet=False, **a)
        fmt = request.args.get("fmt", "csv")
        if fmt == "json":
            body = json.dumps({"query": a["query"], "total": res["total"],
                               "results": [{k: r[k] for k in ("doc_id", "title", "type", "year", "authors", "score")}
                                           for r in res["results"]]}, indent=2)
            return Response(body, mimetype="application/json",
                            headers={"Content-Disposition": 'attachment; filename="results.json"'})
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "type", "year", "title", "authors", "score"])
        for r in res["results"]:
            w.writerow([_csv_safe(r["doc_id"]), r["type"], r["year"] or "", _csv_safe(r["title"]),
                        _csv_safe("; ".join(r["authors"])), r["score"]])
        return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="results.csv"'})

    @app.post("/load-samples")
    def load_samples():
        import sample_data
        n = 0
        for s in sample_data.documents():
            try:
                store.delete_document(s["doc_id"])
            except dr.NotFound:
                pass
            store.add_document(**s)
            n += 1
        return redirect(url_for("home", msg=f"Loaded {n} sample documents."))

    # ---- one document ---------------------------------------------------------------
    @app.get("/doc/<doc_id>")
    def doc(doc_id):
        try:
            page = int(request.args.get("page", 1))
        except ValueError:
            page = 1
        d = store.get_document(doc_id, with_text=False)
        reading = store.read(doc_id, page=max(page, 1), page_size=PAGE_CHARS)
        return render_template("doc.html", d=d, r=reading, related=store.related(doc_id, 5),
                               insights=store.insights(doc_id), cite_formats=list(CITE_FORMATS),
                               others=[x for x in store.list_documents(limit=500) if x["_id"] != doc_id])

    @app.get("/doc/<doc_id>/cite/<fmt>")
    def cite(doc_id, fmt):
        if fmt not in CITE_FORMATS:
            raise dr.DocError(f"unknown citation format {fmt!r}; use bibtex, ris or apa")
        ext, mime = CITE_FORMATS[fmt]
        text = store.citation(doc_id, fmt)
        return Response(text, headers={"Content-Type": f"{mime}; charset=utf-8",
                        "Content-Disposition": f'attachment; filename="{re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)}.{ext}"'})

    @app.get("/doc/<doc_id>/file")
    def file(doc_id):
        d = store.get_document(doc_id, with_text=False)
        if d.get("file_id") is None:
            raise dr.NotFound(f"document {doc_id} has no stored file")
        try:
            stream = store.fs.open_download_stream(d["file_id"])
        except Exception:
            raise dr.NotFound(f"the stored file of {doc_id} is missing")
        ctype = d.get("content_type") or "application/octet-stream"
        inline = request.args.get("inline") and ctype in INLINE_TYPES
        name = d["filename"] or doc_id
        disp = f"{'inline' if inline else 'attachment'}; filename=\"{re.sub(r'[^A-Za-z0-9._ -]', '_', name)}\"; filename*=UTF-8''{quote(name)}"

        def chunks():
            while True:
                data = stream.read(1 << 16)
                if not data:
                    break
                yield data
        return Response(chunks(), mimetype=ctype, headers={"Content-Disposition": disp, "Content-Length": str(d["size"])})

    @app.post("/doc/<doc_id>/delete")
    def delete(doc_id):
        r = store.delete_document(doc_id)
        return redirect(url_for("home", msg=f"Deleted {doc_id} ({r['postings_removed']} index entries"
                                            f"{', original file' if r['file_removed'] else ''} removed)."))

    # ---- add ------------------------------------------------------------------------
    @app.route("/add", methods=["GET", "POST"])
    def add():
        if request.method == "GET":
            return render_template("add.html", form={})
        f = request.form
        up = request.files.get("file")
        data = up.read() if up and up.filename else None
        year = (f.get("year") or "").strip()
        try:
            r = store.add_document(
                f.get("title", ""), f.get("type", ""),
                authors=dr._split_multi([f.get("authors", "")], r";|\n"),
                keywords=dr._split_multi([f.get("keywords", "")], r",|\n"),
                year=int(year) if year else None, abstract=f.get("abstract", ""), text=f.get("text", ""),
                file_bytes=data, filename=up.filename if data is not None else None)
        except (dr.DocError, ValueError) as e:
            return render_template("add.html", form=f, error=str(e)), 400
        note = "Document stored."
        twin = store.related(r["doc_id"], 1)
        if twin and twin[0]["similarity"] >= DUP_THRESHOLD:
            note += f" Warning: it is very similar to {twin[0]['doc_id']} ({twin[0]['similarity']:.2f}), possibly a duplicate."
        return redirect(url_for("doc", doc_id=r["doc_id"], msg=note))

    # ---- compare --------------------------------------------------------------------
    @app.get("/compare")
    def compare():
        a, b = request.args.get("a", ""), request.args.get("b", "")
        result = store.compare(a, b, top_terms=14) if a and b else None
        return render_template("compare.html", a=a, b=b, c=result, all_docs=store.list_documents(limit=500))

    # ---- association: graph, duplicates, keywords, analytics ------------------------------
    @app.get("/graph")
    def graph():
        threshold = min(max(_float("threshold", 0.10), 0.0), 1.0)
        doc_type = request.args.get("type") or None
        g = store.similarity_graph(threshold, doc_type)
        order = [d for comp in g["components"] for d in comp]
        shown = order[:MAX_GRAPH_NODES]
        info = {n["id"]: n for n in g["nodes"]}
        pos = {d: {**p, "type": info[d]["type"], "title": info[d]["title"]} for d, p in _circle_layout(shown).items()}
        edges = [e for e in g["edges"] if e["source"] in pos and e["target"] in pos]
        return render_template("graph.html", g=g, pos=pos, edges=edges, threshold=threshold,
                               sel_type=doc_type or "", truncated=len(order) > len(shown), info=info)

    @app.get("/duplicates")
    def duplicates():
        threshold = min(max(_float("threshold", DUP_THRESHOLD), 0.05), 1.0)
        return render_template("duplicates.html", threshold=threshold, pairs=store.find_duplicates(threshold))

    @app.get("/keywords")
    def keywords():
        kw = (request.args.get("keyword") or "").strip()
        rows = store.keyword_cooccurrence(min_count=_int("min") or 1, keyword=kw or None,
                                          doc_type=request.args.get("type") or None, limit=100)
        return render_template("keywords.html", rows=rows, keyword=kw, min_count=_int("min") or 1,
                               sel_type=request.args.get("type", ""))

    @app.get("/analytics")
    def analytics():
        an = store.analytics()
        rows = an["by_year"]
        top = max((r["total"] for r in rows), default=0) or 1
        height, base_y, slot = 200.0, 230.0, (600.0 / max(len(rows), 1))
        bars = []
        for i, r in enumerate(rows):
            x, y, segs = 50 + i * slot + slot * 0.15, base_y, []
            for t in dr.DOC_TYPES:
                if r[t]:
                    h = r[t] / top * height
                    y -= h
                    segs.append({"type": t, "y": round(y, 1), "h": round(h, 1), "n": r[t]})
            bars.append({"x": round(x, 1), "w": round(slot * 0.7, 1), "year": r["year"], "total": r["total"],
                         "top": round(y, 1), "segs": segs})
        ticks = sorted({0, top} | ({top // 2} if top >= 4 else set()))
        counts = dict(an["top_authors"])
        names = sorted({n for e in an["author_edges"] for n in (e["a"], e["b"])} | {n for n, _ in an["top_authors"][:8]})[:30]
        pos = _circle_layout(names, r=230)
        net_edges = [e for e in an["author_edges"] if e["a"] in pos and e["b"] in pos]
        return render_template("analytics.html", an=an, bars=bars, ticks=ticks, top=top, base_y=base_y, height=height,
                               pos=pos, net_edges=net_edges, counts=counts)

    @app.get("/stats")
    def stats():
        return render_template("stats.html", s=store.stats())

    # ---- JSON API -------------------------------------------------------------------
    def public(d: dict) -> dict:
        return {"id": d["_id"], "title": d["title"], "type": d["type"], "year": d["year"], "authors": d["authors"],
                "keywords": d["keywords"], "abstract": d.get("abstract", ""), "has_file": d.get("file_id") is not None,
                "filename": d.get("filename"), "content_type": d.get("content_type"), "size": d.get("size"),
                "sha256": d.get("sha256"), "added_at": d["added_at"].isoformat() if d.get("added_at") else None,
                "url": url_for("doc", doc_id=d["_id"], _external=True),
                "download": url_for("file", doc_id=d["_id"], _external=True) if d.get("file_id") is not None else None}

    @app.get("/api")
    def api_docs():
        return render_template("api.html", base=request.url_root.rstrip("/"))

    @app.get("/api/search")
    def api_search():
        a = search_args()
        limit = min(max(_int("limit") or 10, 1), 100)
        res = store.search_ex(limit=limit, offset=max(_int("offset") or 0, 0), **a)
        return jsonify({"query": a["query"], "interpreted": res["query"], "total": res["total"],
                        "suggestion": res["suggestion"], "results": res["results"],
                        "facets": {k: [{"value": v, "count": n} for v, n in vs] for k, vs in res["facets"].items()}})

    @app.get("/api/doc/<doc_id>")
    def api_doc(doc_id):
        d = store.get_document(doc_id, with_text=bool(request.args.get("text")))
        out = public(d)
        if request.args.get("text"):
            out["text"] = d["text"]
        return jsonify(out)

    @app.get("/api/related/<doc_id>")
    def api_related(doc_id):
        return jsonify({"doc_id": doc_id, "related": store.related(doc_id, min(max(_int("limit") or 5, 1), 25))})

    @app.get("/api/compare")
    def api_compare():
        return jsonify(store.compare(request.args.get("a", ""), request.args.get("b", "")))

    @app.get("/api/insights/<doc_id>")
    def api_insights(doc_id):
        return jsonify(store.insights(doc_id))

    @app.get("/api/explain/<doc_id>")
    def api_explain(doc_id):
        return jsonify(store.explain(request.args.get("q", ""), doc_id))

    @app.get("/api/stats")
    def api_stats():
        return jsonify(store.stats())

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Web UI for the document retrieval system")
    p.add_argument("--uri", default=os.environ.get("MONGO_URI", dr.DEFAULT_URI))
    p.add_argument("--db", default=os.environ.get("DOCRETRIEVAL_DB", dr.DEFAULT_DB))
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only on a network you trust: there is no login")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args(argv)
    try:
        store = dr.DocumentStore(a.uri, a.db)
    except dr.ServerSelectionTimeoutError:
        print(f"error: cannot reach MongoDB at {a.uri} (see README, section 2).")
        return 2
    print(f"Document retrieval web UI: http://{a.host}:{a.port}  (database '{a.db}')")
    create_app(store).run(host=a.host, port=a.port, debug=a.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
