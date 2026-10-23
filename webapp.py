#!/usr/bin/env python3
"""Web front end for the document retrieval system (Flask).

    python webapp.py                       # http://127.0.0.1:5000, database 'docretrieval'
    python webapp.py --db docretrieval_demo

The page code only calls DocumentStore from docretrieval.py; all data stays in MongoDB/GridFS.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from urllib.parse import quote

from flask import Flask, Response, abort, redirect, render_template, request, url_for
from markupsafe import Markup, escape

import docretrieval as dr

PAGE_CHARS = 1000
MAX_GRAPH_NODES = 80
# Served inline in the browser; everything else is forced to download so an uploaded
# HTML/SVG file can never run script on this site's origin.
INLINE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf", "text/plain"}


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
        return render_template("error.html", message=str(e), code=code), code

    @app.errorhandler(413)
    def too_big(_):
        return render_template("error.html", message="That file is larger than the 64 MB upload limit.", code=413), 413

    @app.context_processor
    def common():
        return {"db_name": store.db.name, "doc_types": dr.DOC_TYPES, "msg": request.args.get("msg"), "read_only": read_only}

    # ---- search / browse ----------------------------------------------------------
    @app.get("/")
    def home():
        q = (request.args.get("q") or "").strip()
        form = {"q": q, "type": request.args.get("type", ""), "year_from": request.args.get("year_from", ""),
                "year_to": request.args.get("year_to", ""), "author": (request.args.get("author") or "").strip(),
                "all": bool(request.args.get("all"))}
        results = docs = None
        if q:
            results = store.search(q, doc_type=form["type"] or None, year_from=_int("year_from"),
                                   year_to=_int("year_to"), author=form["author"] or None,
                                   limit=_int("limit") or 20, require_all=form["all"])
        else:
            docs = store.list_documents(form["type"] or None, limit=500)
        return render_template("index.html", form=form, results=results, docs=docs)

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
                               others=[x for x in store.list_documents(limit=500) if x["_id"] != doc_id])

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
        return redirect(url_for("doc", doc_id=r["doc_id"], msg="Document stored."))

    # ---- compare --------------------------------------------------------------------
    @app.get("/compare")
    def compare():
        a, b = request.args.get("a", ""), request.args.get("b", "")
        result = store.compare(a, b, top_terms=14) if a and b else None
        return render_template("compare.html", a=a, b=b, c=result, all_docs=store.list_documents(limit=500))

    # ---- association: graph + keywords -----------------------------------------------
    @app.get("/graph")
    def graph():
        threshold = min(max(_float("threshold", 0.10), 0.0), 1.0)
        doc_type = request.args.get("type") or None
        g = store.similarity_graph(threshold, doc_type)
        order = [d for comp in g["components"] for d in comp]
        shown = order[:MAX_GRAPH_NODES]
        info = {n["id"]: n for n in g["nodes"]}
        pos = {}
        for i, d in enumerate(shown):
            ang = 2 * math.pi * i / max(len(shown), 1) - math.pi / 2
            pos[d] = {"x": 360 + 250 * math.cos(ang), "y": 330 + 250 * math.sin(ang),
                      "lx": 360 + 280 * math.cos(ang), "ly": 330 + 280 * math.sin(ang) + 4,
                      "anchor": "middle" if abs(math.cos(ang)) < 0.2 else ("start" if math.cos(ang) > 0 else "end"),
                      "type": info[d]["type"], "title": info[d]["title"]}
        edges = [e for e in g["edges"] if e["source"] in pos and e["target"] in pos]
        return render_template("graph.html", g=g, pos=pos, edges=edges, threshold=threshold,
                               sel_type=doc_type or "", truncated=len(order) > len(shown), info=info)

    @app.get("/keywords")
    def keywords():
        kw = (request.args.get("keyword") or "").strip()
        rows = store.keyword_cooccurrence(min_count=_int("min") or 1, keyword=kw or None,
                                          doc_type=request.args.get("type") or None, limit=100)
        return render_template("keywords.html", rows=rows, keyword=kw, min_count=_int("min") or 1,
                               sel_type=request.args.get("type", ""))

    @app.get("/stats")
    def stats():
        return render_template("stats.html", s=store.stats())

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
