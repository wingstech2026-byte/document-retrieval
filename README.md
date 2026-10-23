# Experiment 5: Large-scale document retrieval system (MongoDB + GridFS)

A document store and search engine for four kinds of documents: **papers, patents, technical reports and engineering drawings**.
Metadata and text live in MongoDB collections, the original files (including binary drawings) live in GridFS, and search runs on an
inverted index that is itself stored in MongoDB.

| Required function | Where |
|---|---|
| document storage | `add` / `DocumentStore.add_document` |
| document retrieval | `search` (BM25, filters, snippets), `list` |
| document reading | `read` (paginated) |
| document comparison | `compare` (TF-IDF cosine, shared/distinctive terms, metadata overlap) |
| document association analysis | `related`, `graph`, `cooccur` |
| document download | `download` (GridFS to a folder) |
| extras | `delete` (cleans index and GridFS), `stats`, `demo` |

Files: `docretrieval.py` (engine and CLI), `sample_data.py` (10 original sample documents), `test_docretrieval.py` (54 tests),
`webapp.py` + `app.py` + `templates/` (web site, sections 6 and 7), `test_webapp.py`, `test_features.py`, `test_webapp_features.py`
(tests for the web site and the advanced features, section 8), `requirements.txt`.

## 1. Design

### Collections

```
documents   one record per document
postings    inverted index: one record per (term, doc_id)
meta        one record {_id: "corpus", n, total_len}: document count and total length, used by BM25
fs.files, fs.chunks   GridFS: the original files
```

`documents` record:

```json
{
  "_id": "pat-001",                 // generated as <pap|pat|rep|drw>-<8 hex> unless --doc-id is given
  "title": "Method for Compressing Posting Lists ...",
  "type": "patent",                 // paper | patent | report | drawing
  "authors": ["Sofia Alvarez"],
  "keywords": ["inverted index", "compression", "posting list"],   // lower-cased, de-duplicated
  "year": 2020,
  "abstract": "...",
  "text": "...",                    // full text (given, or extracted from a .txt/.pdf file); drawings use their description
  "length": 192,                    // weighted token count, used by BM25 length normalisation
  "file_id": ObjectId("..."),       // -> fs.files._id (null if no file)
  "filename": "pat-001.txt", "content_type": "text/plain", "size": 1457, "sha256": "...",
  "added_at": ISODate("...")
}
```

`postings` record: `{ "term": "index", "doc_id": "pat-001", "tf": 6 }`.
`tf` is weighted: a term occurrence counts 3 in the title, 2 in a keyword, 1 in the abstract or body.

GridFS file metadata: `{ "doc_id": "pat-001", "content_type": "...", "doc_type": "patent" }`. Drawings are real binary files
(the samples are generated PNGs); the engine never tries to decode them.

### Indexes

| Collection | Index | Purpose |
|---|---|---|
| postings | `(term, doc_id)` **unique** | one record per pair; find all postings of a term; ordered scan for the graph |
| postings | `doc_id` | rebuild a document's vector (compare/related) and delete its entries |
| documents | `(type, year)`, `year`, `authors`, `keywords` | metadata filters, listing, co-occurrence |

### Algorithms

* **Tokenising**: lower-case, split on non-alphanumerics, drop stop words and 1-letter tokens, strip plural endings (`networks` ->
  `network`, `gearboxes` -> `gearbox`). There is no further stemming, so `optimize` and `optimization` are different terms.
* **Search (BM25)**: for each query term the postings are fetched with one `$in` query, and a document scores
  `sum idf(t) * tf * (k1+1) / (tf + k1 * (1 - b + b * dl/avgdl))` with `idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, `k1 = 1.5`, `b = 0.75`.
  Any query term may match (`--all` requires every term). Filters on type, year range and author (case-insensitive substring)
  are applied to the candidate documents. The snippet is the 28-word window with the most query-term hits, with hits marked `**like this**`.
* **Reading**: the text is cut into pages of a chosen number of characters, preferring a paragraph break, then a word break.
  The pages concatenate back to the exact original text.
* **Comparison**: TF-IDF vectors from the two documents' postings, weight `(1 + ln tf) * (ln((N+1)/(df+1)) + 1)`, then cosine
  similarity. Shared terms are ranked by their joint weight (their contribution to the dot product), distinctive terms by their weight
  in their own document. Metadata overlap: shared authors, shared keywords, Jaccard of each, same type, year gap.
* **Related documents**: take the 25 heaviest terms of the document, collect candidate documents from those postings,
  and score up to 200 candidates exactly by cosine.
* **Similarity graph**: all-pairs cosine, computed term by term from the postings in index order. Edges are pairs at or above the
  threshold; connected components give topic groups. Output can be saved as Graphviz (`--dot`) or JSON (`--json`).
* **Keyword co-occurrence**: one MongoDB aggregation (`$unwind` twice, keep `a < b`, `$group` by pair), so counting happens in the database.
* **Delete**: remove the document's postings, then its GridFS file, then the document record and the corpus counters, in that order.
  A half-finished delete never leaves a searchable ghost and can simply be run again.
* **Add** writes the GridFS file, the document and the postings. If a step fails, everything written so far is undone.
* **Download** streams the file out of GridFS into a folder, never overwrites (a numeric suffix is added: `name (1).ext`),
  keeps only the base name of the stored file name, and verifies the SHA-256 recorded at storage time.

### Limits worth knowing

* A standalone `mongod` has no multi-document transactions, so add/delete use ordered writes and manual rollback as described above
  rather than a transaction (use a replica set and `with_transaction` if you need that).
* `similarity_graph` costs about the sum of `df^2` over all terms, so it suits thousands of documents. For a much larger corpus use
  `related`, or restrict the graph with `--type`. Search fetches all postings of the query terms; very common terms are the expensive ones.
* Text is extracted only for plain-text files and PDFs (`pypdf`, optional). Scanned PDFs and drawings are indexed through their
  title, keywords and abstract/description.

## 2. MongoDB setup

Any MongoDB 5.0+ works. The code was tested against MongoDB **8.0.34** (Community) on Windows.

**Docker** (the quickest):

```bash
docker run -d --name mongo-docs -p 27017:27017 -v mongo-docs-data:/data/db mongo
```

**Windows without Docker** (what was used for the test run): download the Community `.zip` from
<https://www.mongodb.com/try/download/community> (or install the MSI), unzip it, then

```powershell
mkdir C:\mongo-data
.\mongodb-win32-x86_64-windows-8.0.34\bin\mongod.exe --dbpath C:\mongo-data --bind_ip 127.0.0.1 --port 27017
```

**macOS**: `brew tap mongodb/brew && brew install mongodb-community && brew services start mongodb-community`.
**Linux**: install `mongodb-org` from the MongoDB repository and `sudo systemctl start mongod`.

Then install the Python packages:

```bash
pip install -r requirements.txt        # add: pip install -r requirements-dev.txt  for the tests
```

The default connection is `mongodb://localhost:27017`. Override it with `--uri ...` or `$MONGO_URI`; choose the database with
`--db NAME` or `$DOCRETRIEVAL_DB` (default `docretrieval`; the demo uses `docretrieval_demo`). Indexes are created automatically on first connect.
If the server cannot be reached the CLI says so and exits with status 2.

## 3. Running it

```bash
python docretrieval.py demo                    # load the samples and tour every feature
python docretrieval.py -h                      # all subcommands

python docretrieval.py add --title "Notes on shear stress" --type report \
       --author "A. Rivera; B. Chen" --keyword "shear, torsion" --year 2024 --file notes.pdf
python docretrieval.py list --type patent
python docretrieval.py search inverted index compression --limit 5
python docretrieval.py search vibration --type report --year-from 2023 --author berg
python docretrieval.py search bearing vibration --all
python docretrieval.py read pap-003 --page 2 --page-size 450
python docretrieval.py compare pap-003 rep-002
python docretrieval.py related pat-002 --limit 4
python docretrieval.py graph --threshold 0.10 --dot graph.dot --json graph.json
python docretrieval.py cooccur --min-count 2          # or --keyword "wind turbine"
python docretrieval.py download drw-001 --dest downloads
python docretrieval.py delete pap-001 pap-002
python docretrieval.py stats
```

`add` takes `--type paper|patent|report|drawing` (`technical report` is accepted for `report`), `--text` or `--text-file` for the body,
and `--file` for the original (PDF, TXT, PNG, DXF, any binary). If no text is given and the file is a `.txt` or `.pdf`, the text is extracted from it.
The `demo` command only touches the ten sample ids in its own database: it deletes and re-adds those, so it can be run repeatedly.

From Python:

```python
from docretrieval import DocumentStore
st = DocumentStore("mongodb://localhost:27017", "docretrieval")
st.add_document("My report", "report", authors=["Me"], keywords=["x"], year=2024, file_path="report.pdf")
st.search("vibration", doc_type="paper", year_from=2022, author="berg")
```

## 4. Tests

```bash
python -m pytest -q          # needs MongoDB; each run uses a throw-away database that is dropped afterwards
```

54 tests (about 10 s), grouped as: storage and index structure (unique `(term, doc_id)`, rollback on bad input), ranking (a hand-computed
BM25 check, rare-term weighting, title/keyword matching, `--all`, snippets), filters (type, year range, author, combined), reading
(lossless pagination, out-of-range pages), comparison (identity = 1, symmetry, shared/distinctive terms, metadata overlap),
association analysis (related, graph vs. `compare`, threshold monotonicity, component grouping, co-occurrence vs. a manual count),
download round trip (byte-identical, binary PNG, no overwrite, path traversal, corruption detected) and delete cleanup (no residue
in `documents`, `postings`, `fs.files`, `fs.chunks`; counters; search/analysis no longer see it; resumable after a partial failure).
If MongoDB is not reachable the tests are skipped, not passed.

Result of the last run against a local `mongod` 8.0.34: `54 passed in 10.13s`. The tests were also checked by breaking the code on
purpose (skipping posting deletion, and removing BM25 length normalisation); the relevant tests failed each time.

## 5. Example output (from `python docretrieval.py demo`)

Abridged; the full run also shows a listing, a second comparison, per-step statistics and the delete cleanup.

```
=== 1. Storage: loading 10 sample documents into database 'docretrieval_demo' ===
  stored pap-001  paper    Adaptive Learned Indexes for Skewed Key Distributions      file  1,282 B
  ...
  stored drw-001  drawing  Gearbox Housing Assembly Drawing, Sheet 3                  file  1,016 B
database: docretrieval_demo
documents: 10  by type: drawing=2, paper=4, patent=2, report=2
GridFS files: 10 (11,733 bytes)
index: 945 postings, 701 distinct terms, avg doc length 161.7

=== 3. Ranked search (BM25): 'inverted index compression' ===
 1. [7.436] pat-001  patent, 2020  Method for Compressing Posting Lists Using Block-wise Delta Coding
     authors: Sofia Alvarez
     ...storing a posting list of an **inverted** **index**, the posting list comprising document identifiers in ascending order...
 2. [4.732] pap-004  paper, 2021  Ranking Short Technical Documents with Sparse and Dense Signals
     ...words are strong evidence, and the **inverted** **index** that supports it is cheap to build and update. ...
 3. [2.420] pap-001  paper, 2022  Adaptive Learned Indexes for Skewed Key Distributions
     Classic B-tree **indexes** spend most of their lookup time walking from the root to a leaf. A learned **index** shortcuts ...

=== 3c. Filtered search: 'nosql cluster', years 2021-2023, author 'haddad' ===
 1. [3.006] rep-001  report, 2023  Load Test Report: Replica Set Failover in a Three-Node Document Cluster
 2. [2.429] pap-002  paper, 2023  Sharding Strategies for Document Stores under Hotspot Workloads
 3. [0.844] pap-001  paper, 2022  Adaptive Learned Indexes for Skewed Key Distributions

=== 4. Reading (paginated, 450 characters per page): pap-003, page 2 ===
Federated Training of Fault Detectors on Wind Turbine Gearboxes  [pap-003, paper, 2024]
original file: pap-003.txt (use 'download pap-003' to fetch it)
--- page 2 of 3 ---
We therefore train a fault detector in a federated way. Each farm fits a compact neural network ...

=== 5. Comparison ===
A: pap-003  Federated Training of Fault Detectors on Wind Turbine Gearboxes (paper, 2024)
B: rep-002  Condition Monitoring Field Report: Gearbox Bearings at Site K (report, 2024)
TF-IDF cosine similarity: 0.2262
Shared terms (25 total, top by joint weight): gearbox, turbine, fault, vibration, detector, wind, detection, site, bearing, gear
Only in A: farm, federated, train, training, data, model, raw, learning, yet, window
Only in B: monitoring, condition, baseline, trend, race, finding, eight, spectra, day, outer
Metadata: same type=False, year gap=0
  shared authors (0.33 Jaccard): Tomas Berg
  shared keywords (0.43 Jaccard): fault detection, wind turbine, vibration
(pap-003 vs pat-001, an unrelated pair, scores 0.0305)

=== 6b. Similarity graph (cosine >= 0.10) ===
threshold 0.1: 10 nodes, 11 edges
  pap-003 -- rep-002  0.2262
  pat-002 -- rep-002  0.2050
  ...
connected groups with 2+ documents: 3
  group 1: drw-001, pap-003, pat-002, rep-002       <- wind-turbine condition monitoring
  group 2: drw-002, pap-001, pap-002, rep-001       <- document databases
  group 3: pap-004, pat-001                         <- inverted-index retrieval

=== 6c. Keyword co-occurrence (pairs seen in 2+ documents) ===
  3  gearbox + wind turbine   (drw-001, pap-003, pat-002)
  3  vibration + wind turbine   (pap-003, pat-002, rep-002)
  2  information retrieval + inverted index   (pap-004, pat-001)
  2  mongodb + nosql   (pap-002, rep-001)

=== 7. Download an engineering drawing from GridFS ===
wrote ...\downloads\gearbox_housing_sheet3.png (1,016 bytes, sha256 e9a72661ab92c234..., checksum verified)

=== 8. Delete cleanup: add a throwaway document with a file, then delete it ===
after adding tmp-demo: 11 docs, 951 postings, 11 GridFS files
deleted tmp-demo: 6 postings removed, GridFS file removed
after deleting it : 10 docs, 945 postings, 10 GridFS files
postings still pointing at tmp-demo: 0
```

## 6. Web site

A Flask app on top of the same engine; every function above is available in the browser.

```bash
pip install -r requirements.txt            # includes flask
python webapp.py --db docretrieval_demo    # then open http://127.0.0.1:5000
```

On an empty database the home page offers a "Load the 10 sample documents" button. Options: `--uri`, `--db`, `--host`, `--port`, `--debug`
(`$MONGO_URI` and `$DOCRETRIEVAL_DB` work too).

| Page | What it does |
|---|---|
| `/` Search | BM25 search box with type, year-range and author filters, an "all terms" switch, and highlighted snippets; with no query it lists every document |
| `/doc/<id>` | metadata, paginated reading (1,000 characters per page), image preview for drawings, **Download original**, **Delete**, "Compare with", related documents |
| `/compare` | pick two documents: cosine bar, shared and distinctive terms, author/keyword overlap |
| `/graph` | similarity graph (SVG, nodes coloured by type) with a threshold control, topic groups and a link table |
| `/keywords` | keyword co-occurrence, optionally for one keyword or one type |
| `/add` | upload form: metadata, text and any original file (up to 64 MB) |
| `/stats` | counts by type, index size, GridFS usage, top keywords and terms |

Notes: downloads stream from GridFS to the browser; only images, PDF and plain text are ever displayed inline, anything else (for example an
uploaded `.html`) is forced to download. All stored text is HTML-escaped. Delete is a POST with a confirmation dialog and removes the document,
its index entries and its GridFS file. **There is no login**: the server listens on 127.0.0.1 by default, so do not expose it to an untrusted
network as is. It uses Flask's development server, which is fine for a lab or demo.

`python -m pytest -q` runs 127 tests (see section 8); the first 78 are (54 engine + 24 web: every page renders, filters and highlighting, paging, byte-identical file download,
no inline active content, HTML escaping, upload then search then download then delete, validation errors, repeatable sample loading, empty database).

## 7. Deploying the web site on Vercel

`app.py` is the entry point Vercel looks for: it exposes the Flask application as `app`, and the stylesheet is inlined into the pages, so no
static-file setup is needed. Vercel cannot reach a MongoDB running on your own computer, so you need a hosted one, for example a free
**MongoDB Atlas** cluster.

1. **Atlas**: create a free cluster; add a database user (username and password); under *Network Access* allow `0.0.0.0/0`
   (Vercel's IP addresses change, so a fixed allow-list does not work); copy the connection string
   (`mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/`).
2. **Load the data from your own computer** (this also creates the indexes):
   ```bash
   python docretrieval.py --uri "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/" --db docretrieval demo
   ```
   The demo loads the 10 sample documents into the database named by `--db` (here `docretrieval`; without `--db` it uses `docretrieval_demo`).
   Use `add` with your own files for real data.
3. **Vercel**: import the GitHub repository, then in *Settings, Environment Variables* add
   `MONGO_URI` (the connection string) and `DOCRETRIEVAL_DB` (the same database name as step 2), and redeploy.
4. If something is missing or wrong, the site shows a "Setup needed" page that says what (no `MONGO_URI`, or cannot connect).

**The site is read-only by default.** There is no login, so on a public URL anyone could otherwise upload or delete documents. With
`READ_ONLY=1` (the default) the Add page, Delete buttons and sample loader are removed and every write request is refused (HTTP 403);
searching, reading, comparing, the graph, keywords, stats and downloads all work. Set `READ_ONLY=0` only for a private deployment.

Vercel limits to know about: request bodies are capped at about 4.5 MB (so large uploads would fail even with writes enabled), and each page
request is a short-lived function, so the all-pairs similarity graph is meant for small corpora (it is instant for the sample data).

## 8. Advanced features

These are in the engine (`docretrieval.py`), the web site and the command line.

**Query language** (works in the search box, `search` CLI command, `/api/search` and exports):

| Write | Meaning |
|---|---|
| `"posting list"` | the words must appear next to each other, in that order |
| `vibration -sensor` | leave out documents containing *sensor* |
| `type:patent`, `author:berg`, `author:"tomas berg"` | document type; author name contains |
| `keyword:"wind turbine"`, `title:gearbox` | has this keyword; word is in the title |
| `year:2022`, `year:2020..2023`, `year:>=2021`, `year:<2020` | publication year |

Filters can be used on their own (`type:patent year:>2020` lists matching documents) or together with words, and they combine with the
Type/Year/Author form fields. Phrases are checked against the stored text of the best candidates, so very large result sets cap at 2000 candidates.

| Feature | Web | CLI | What it does |
|---|---|---|---|
| Facets | sidebar on the search page | `search` prints counts | counts by type, year, author and keyword over **all** matches; clicking one adds the filter to the query |
| Paging, sorting | Previous/Next, "Sort by" | `--offset`, `--sort year` | 10 results per page; relevance or newest first |
| Did you mean | banner above the results | `search` | an indexed word within edit distance 1 (2 for long words) replaces an unknown word; the commoner term wins |
| Score explanation | "why this score?" link, `/explain/<id>?q=...` | `explain DOC QUERY` | the BM25 score split into one line per term (tf, df, idf, contribution), with the title/keyword weights |
| Document insights | "At a glance" card | `insights DOC` | key terms by TF-IDF, a 3-sentence extractive summary (sentences scored by the weight of their terms, kept in reading order), reading time, frequent terms missing from the keywords |
| Analytics | `/analytics` | `authors` | documents per year by type, most active authors, top keywords, co-authorship network (author pairs counted by a MongoDB aggregation) |
| Near-duplicates | `/duplicates`, and a warning after **Add** | `duplicates --threshold 0.6` | pairs above a cosine threshold; adding a document at 0.6 or more similar to an existing one shows a warning |
| Citations | BibTeX / RIS / APA links on each document | `cite DOC --format bibtex` | reference text, with BibTeX special characters escaped |
| Export | CSV / JSON links on the results | | up to 1000 results; CSV cells that start with `= + - @` are prefixed with `'` so spreadsheets do not run them |
| JSON API | `/api` lists the endpoints | | `/api/search`, `/api/doc/<id>`, `/api/related/<id>`, `/api/compare`, `/api/insights/<id>`, `/api/explain/<id>`, `/api/stats`; errors are JSON |

Examples:

```bash
python docretrieval.py search '"posting list"'
python docretrieval.py search vibration -federated type:report year:">=2023"
python docretrieval.py search gearbxo vibraton            # prints: Did you mean: gearbox vibration
python docretrieval.py explain pat-002 vibration sensor
python docretrieval.py insights pap-003
python docretrieval.py cite pap-003 --format ris
python docretrieval.py authors
python docretrieval.py duplicates --threshold 0.2
curl "http://127.0.0.1:5000/api/search?q=vibration+-federated&limit=3"
```

(In PowerShell and bash, quote `year:">=2023"` or the shell treats `>` as a redirect.) `python docretrieval.py demo` ends with a section 9 that
runs these features on the sample data.

Tests: `test_features.py` (28 engine tests: the parser and each operator, phrases, exclusion, facets against the full result set, paging,
spelling, explain adding up to the real search score, insights, analytics against manual counts, citations, duplicates) and
`test_webapp_features.py` (21 web tests: facet links, suggestions, pagination, exports including CSV-formula safety, citation downloads,
analytics and duplicate pages, the add-time warning, the JSON API and its errors, and that the read-only site keeps all of this).
Breaking the phrase check, the exclusion or the spelling distance on purpose makes the matching test fail.
