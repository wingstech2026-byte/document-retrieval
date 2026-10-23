"""Regenerates the code, data and terminal images used by the presentation (docs/code/*.png).

    python docs/make_images.py

Code images are cut from the real source files with their real line numbers; data images come from
the demo database (docretrieval_demo, created by `python docretrieval.py demo`); terminal images are
the real output of the command-line tool.
"""
import json
import os
import subprocess
import sys
import textwrap

from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import JsonLexer, PythonLexer, TextLexer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "code")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)


def render(code: str, name: str, lexer, start: int | None = None, size: int = 24):
    fmt = ImageFormatter(style="monokai", font_name="Consolas", font_size=size, image_pad=22, line_pad=6,
                         line_numbers=start is not None, line_number_start=start or 1,
                         line_number_bg="#272822", line_number_fg="#75715e", line_number_separator=False, line_number_pad=14)
    with open(os.path.join(OUT, name + ".png"), "wb") as fh:
        fh.write(highlight(code, lexer, fmt))
    print("wrote", name)


def source(path: str, first: int, last: int) -> str:
    lines = open(os.path.join(ROOT, path), encoding="utf-8").read().split("\n")[first - 1:last]
    return textwrap.dedent("\n".join(lines))


def code(name, path, first, last, size=24):
    render(source(path, first, last), name, PythonLexer(), start=first, size=size)


# ---- code ---------------------------------------------------------------------
code("c01_tokenize_index_terms", "docretrieval.py", 91, 113)
code("c02_indexes", "docretrieval.py", 348, 354)
code("c03_add_document_write", "docretrieval.py", 431, 458, size=22)
code("c04a_fetch_postings", "docretrieval.py", 532, 541)
code("c04b_bm25_scoring", "docretrieval.py", 553, 560)
code("c05a_tfidf_compare", "docretrieval.py", 670, 681)
code("c05b_cosine", "docretrieval.py", 200, 206)
code("c06_aggregation_pipeline", "docretrieval.py", 783, 796)
code("c07_delete_document", "docretrieval.py", 476, 490)
code("c08_download_stream", "docretrieval.py", 909, 922)
code("c09_vercel_entry", "app.py", 29, 37, size=22)

# ---- real records stored in MongoDB -------------------------------------------------
import docretrieval as dr  # noqa: E402

st = dr.DocumentStore(db_name="docretrieval_demo")
d = st.documents.find_one({"_id": "pat-001"})
d["text"] = d["text"][:90] + "..."
d["abstract"] = d["abstract"][:90] + "..."
d["added_at"] = d["added_at"].isoformat()
d["file_id"] = f"ObjectId('{d['file_id']}')"
posts = [{k: v for k, v in p.items() if k != "_id"} for p in st.postings.find({"doc_id": "pat-001", "term": {"$in": ["posting", "inverted", "compression"]}}).sort("term")]
fs_file = st.db["fs.files"].find_one({"filename": "gearbox_housing_sheet3.png"})
chunk = st.db["fs.chunks"].find_one({"files_id": fs_file["_id"]})
fs_view = {"_id": f"ObjectId('{fs_file['_id']}')", "filename": fs_file["filename"], "length": fs_file["length"],
           "chunkSize": fs_file["chunkSize"], "uploadDate": fs_file["uploadDate"].isoformat(), "metadata": fs_file["metadata"]}
chunk_view = {"files_id": f"ObjectId('{fs_file['_id']}')", "n": chunk["n"], "data": f"<Binary: {len(chunk['data'])} bytes, PNG signature {bytes(chunk['data'][:4])!r}>"}
meta_doc = st.meta.find_one({"_id": "corpus"})
render("// db.documents.findOne({_id: \"pat-001\"})\n" + json.dumps(d, indent=2), "d01_documents_record", JsonLexer(), size=22)
render("// db.postings.find({doc_id: \"pat-001\", term: {$in: [...]}})\n" + "\n".join(json.dumps(p) for p in posts)
       + "\n\n// db.meta.findOne({_id: \"corpus\"})\n" + json.dumps(meta_doc), "d02_postings_records", JsonLexer(), size=24)
render("// db.fs.files.findOne({filename: \"gearbox_housing_sheet3.png\"})\n" + json.dumps(fs_view, indent=2)
       + "\n\n// db.fs.chunks.findOne({files_id: ...})\n" + json.dumps(chunk_view, indent=2), "d03_gridfs_records", JsonLexer(), size=22)
lines = ["// db.postings.getIndexes()   and   db.documents.getIndexes()"]
for c in ("postings", "documents"):
    lines.append(f"{c}")
    for i in st.db[c].list_indexes():
        keys = ", ".join(f"{k}: {v}" for k, v in i["key"].items())
        lines.append(f"  {i['name']:<17}{'UNIQUE ' if i.get('unique') else '       '}{{{keys}}}")
render(chr(10).join(lines), "d04_indexes", TextLexer(), size=22)
st.close()


# ---- real terminal output -----------------------------------------------------------
def run(args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", DOCRETRIEVAL_DB="docretrieval_demo")
    p = subprocess.run([sys.executable, "docretrieval.py"] + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
    return p.stdout.rstrip()


def term(name, cmd, args, trim=None, size=22):
    out = run(args)
    lines = out.split("\n")
    if trim:
        lines = lines[:trim]
    render(f"$ {cmd}\n" + "\n".join(x[:118] for x in lines), name, TextLexer(), size=size)


term("t01_cli_search", 'python docretrieval.py search "inverted index" compression --limit 2', ["search", "inverted", "index", "compression", "--limit", "2"], trim=11)
term("t02_cli_compare", "python docretrieval.py compare pap-003 rep-002", ["compare", "pap-003", "rep-002"])
term("t03_cli_graph", "python docretrieval.py graph --threshold 0.10", ["graph", "--threshold", "0.10"])
term("t04_cli_cooccur", "python docretrieval.py cooccur --min-count 2", ["cooccur", "--min-count", "2"])
term("t05_cli_explain", "python docretrieval.py explain pat-002 vibration sensor", ["explain", "pat-002", "vibration", "sensor"])
demo = subprocess.run([sys.executable, "docretrieval.py", "demo"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                      env=dict(os.environ, PYTHONIOENCODING="utf-8")).stdout
sec = demo[demo.index("=== 8."):demo.index("=== 9.")].rstrip()
render("$ python docretrieval.py demo      # step 8\n" + sec, "t06_delete_cleanup", TextLexer(), size=22)
lines = ["$ python -m pytest -q <file>        # real results, one run per test file"]
total = 0
for f in ("test_docretrieval.py", "test_webapp.py", "test_features.py", "test_webapp_features.py"):
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip().splitlines()[-1]
    total += int(out.split()[0])
    lines.append(f"{f:<26}{out}")
lines.append(f"{'total':<26}{total} passed")
render(chr(10).join(lines), "t07_tests", TextLexer(), size=24)
subprocess.run(["rm", "-rf", "downloads"], cwd=ROOT)
