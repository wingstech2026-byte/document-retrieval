# Presentation guide: Large-scale Document Retrieval System

Everything you need to present Experiment 5: how the system works, how the documents are stored, the ER diagram, screenshots of the
running system, the code to explain, a slide-by-slide script, a live-demo plan and answers to likely questions.

| File | What it is |
|---|---|
| [`Experiment5_Presentation.pptx`](Experiment5_Presentation.pptx) | The slide deck: 24 slides, with speaker notes on every slide (View → Notes in PowerPoint) |
| `PRESENTATION_GUIDE.md` | This guide |
| `diagrams/` | ER diagram, architecture and the two flow diagrams as PNG files |
| `screenshots/` | Screenshots of the website |
| `code/` | Images of the code (with real line numbers), of the stored records and of terminal output |
| `make_images.py` | Regenerates the images in `code/` from the real source and database |

All numbers below come from the demo database (10 sample documents). Re-create it with `python docretrieval.py demo`.

---

## 1. The project in one minute

**What it is.** A system that stores four kinds of documents (papers, patents, technical reports, engineering drawings) and offers six
functions: storage, retrieval, reading, comparison, association analysis and download. It is built on MongoDB, using the pymongo driver
and GridFS for the original files.

**How you use it.** Three ways, all calling the same Python engine: a website, a command line, and a JSON API.

**The idea that makes search work.** For every word, MongoDB stores a list of the documents that contain it (an *inverted index*, kept as a
collection called `postings`). A search looks up the words of the query in that list, scores the documents with the *BM25* formula, and
returns them ranked. Comparing two documents turns each into a vector of word weights (*TF-IDF*) and measures the angle between them
(*cosine similarity*).

**What I added beyond the assignment.** A query language (`"phrases"`, `-exclusions`, `type:patent`, `year:2020..2023`), facets, spelling
suggestions, a "why this score?" page, automatic summaries, analytics, a near-duplicate finder, citations, CSV/JSON export, a JSON API,
a cloud deployment and 127 automatic tests.

---

## 2. How it works, exactly

### 2.1 The architecture

![Architecture](diagrams/architecture.png)

* `docretrieval.py` contains one class, **`DocumentStore`**. It is the *only* code that talks to MongoDB. It also contains the
  command-line interface (argparse).
* `webapp.py` is a Flask website. It only calls `DocumentStore`: no database code of its own. `app.py` exposes it to Vercel.
* So a feature is written once and works on the website, in the API and on the command line, and the tests can test the engine directly.
* Data lives in one MongoDB database: collections `documents`, `postings`, `meta`, and GridFS (`fs.files`, `fs.chunks`).

### 2.2 The data model (ER diagram)

![ER diagram](diagrams/er_diagram.png)

The same model in Mermaid (GitHub draws this one):

```mermaid
erDiagram
    DOCUMENTS ||--o{ POSTINGS : "is indexed by (doc_id)"
    DOCUMENTS ||--o| FS_FILES : "keeps original file (file_id)"
    FS_FILES ||--|{ FS_CHUNKS : "is split into (files_id)"
    DOCUMENTS }o--o{ AUTHOR : "written by (embedded list)"
    DOCUMENTS }o--o{ KEYWORD : "tagged with (embedded list)"
    TERM ||--o{ POSTINGS : "occurs in (derived)"
    DOCUMENTS {
        string _id PK "pat-001"
        string title
        string type "paper, patent, report, drawing"
        int year
        string abstract
        string text
        string_list authors "embedded"
        string_list keywords "embedded"
        int length "weighted number of terms"
        ObjectId file_id FK "to fs.files"
        string filename
        int size
        string sha256 "integrity check"
        date added_at
    }
    POSTINGS {
        string term PK "with doc_id, UNIQUE"
        string doc_id PK
        number tf "weighted count"
    }
    FS_FILES {
        ObjectId _id PK
        string filename
        int length
        int chunkSize "261120 = 255 KB"
        date uploadDate
        object metadata "doc_id, content_type, doc_type"
    }
    FS_CHUNKS {
        ObjectId _id PK
        ObjectId files_id FK
        int n "piece number"
        binary data
    }
    META {
        string _id PK "corpus"
        int n "number of documents"
        int total_len "sum of lengths"
    }
```

**How to read it, relationship by relationship:**

| Relationship | Meaning |
|---|---|
| `documents` 1 — 0..N `postings` | A document has one posting per distinct term in it (the drawing `drw-001` has 49). Each posting belongs to exactly one document. |
| `documents` 0..1 — 1 `fs.files` | A document has at most one original file. `file_id` points to it. |
| `fs.files` 1 — 1..N `fs.chunks` | GridFS cuts a file into 255 KB chunks. Small files have one chunk. |
| `documents` N — M authors, keywords | Many documents share an author or keyword. They are **embedded as lists**, not separate collections: they are always read together with the document. |
| `term` 1 — N `postings` | A term occurs in many documents. There is no `terms` collection: the vocabulary is the set of distinct `term` values in `postings`. |
| `meta` | Two counters (number of documents, total length) that BM25 needs. |

**Why embed authors and keywords?** In a document database you embed data that is always read together and rarely changes on its own.
One read returns a whole document, with no joins. The price: renaming an author means touching every document that lists them, which this
system never needs to do. **Why a separate `postings` collection and not an array inside `documents`?** One term can occur in thousands of
documents, so the list would grow without limit, and a search needs "all documents of this term" quickly, which an index on a separate
collection gives directly.

**Keys and indexes:**

| Collection | Index | Why |
|---|---|---|
| `postings` | `(term, doc_id)` **unique** | find all documents of a term; one record per pair |
| `postings` | `doc_id` | rebuild one document's vector; delete its postings |
| `documents` | `(type, year)`, `year`, `authors`, `keywords` | filters and keyword counts |

![The indexes MongoDB reports](code/d04_indexes.png)

### 2.3 How one document is stored (a real example)

Adding the engineering drawing *"Gearbox Housing Assembly Drawing, Sheet 3"* (authors: Tomas Berg; year 2023; keywords gearbox, housing,
assembly, wind turbine, technical drawing; a 1,016-byte PNG file):

![Storing flow](diagrams/flow_store.png)

1. **Validate.** Title not empty, type is one of the four, year plausible, id not already used. If anything is wrong, **nothing** is written.
2. **Get the text.** The text given; else text read from a `.txt`/`.pdf` file; for a drawing, its description is the text.
3. **Analyse.** Lower-case; split on anything that is not a letter or digit; drop stop words and one-letter words; strip plural endings.
   The title becomes `gearbox, housing, assembly, drawing, sheet`.
4. **Weight.** A word in the title counts 3, in a keyword 2, in the abstract or text 1. For this drawing `gearbox` gets
   3 (title) + 2 (keyword) + 1 + 1 (abstract and text) = **7**; `sheet` gets 3 + 1 = **4**. The document's `length` is the sum of all
   weighted counts: **95**.
5. **Write**, in this order: (a) the file into GridFS; (b) the record into `documents`; (c) **49 postings** (one per distinct term) with
   `insert_many`; (d) the counters in `meta` with `$inc`. If any step fails, what was already written is removed again.

What ends up in MongoDB, taken from the real database:

![The documents record](code/d01_documents_record.png)

![Postings and meta records](code/d02_postings_records.png)

![GridFS records](code/d03_gridfs_records.png)

* The record keeps `file_id` (link to GridFS), `size` and `sha256` (the file's fingerprint).
* A posting such as `{"term": "posting", "doc_id": "pat-001", "tf": 7}` says: *this term occurs (weighted) 7 times in this document*.
* `meta` shows `n = 10` documents and `total_len = 1617`, so the average length is 161.7.

### 2.4 How search works

![Search flow](diagrams/flow_search.png)

1. **Parse the query.** `"posting list" -sensor type:patent year:>2019` becomes: terms `posting`, `list`; phrase *posting list*;
   exclude `sensor`; filters type = patent, year ≥ 2020.
2. **Look up.** One query on `postings` with `$in` over all the query terms. It uses the `(term, doc_id)` index and returns, for every
   candidate document, the weighted count of each term; it also gives each term's document frequency *df*.
3. **Score with BM25.** For each document, add up one contribution per query term:

   `contribution = idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × len / avglen))`, with `idf = ln(1 + (N − df + 0.5) / (df + 0.5))`,
   `k1 = 1.5`, `b = 0.75`.

   * **idf**: a word found in few documents is stronger evidence than one found in many.
   * **tf part**: more occurrences help, but with diminishing returns (it approaches `k1 + 1`).
   * **len / avglen**: long documents are slightly penalised so they do not win just by being long.
4. **Filter.** Type, year range, author, keyword, title words, excluded words; the exact phrase is checked in the stored text.
5. **Present.** Rank by score (ties by id), cut a snippet with the matching words highlighted, count facets, return ten per page.
   If nothing matches, look for a spelling suggestion.

**Worked example (real numbers).** Query `vibration sensor`, document `pat-002` (length 167; N = 10; average length 161.7):

| term | count in document | documents with term | idf | contribution |
|---|---|---|---|---|
| vibration | 7 | 4 of 10 | 0.894 | **1.832** |
| sensor | 8 | 5 of 10 | 0.693 | **1.454** |
| | | | **score** | **3.286** |

The website's "why this score?" page shows exactly this table, and a test checks that its total equals the score in the result list.

![Why this score](screenshots/09_explain_score.png)

### 2.5 Reading, comparison, association, download, delete

* **Reading.** The text is cut into pages of about 1,000 characters, at a paragraph break if possible, else at a space, so no word is split.
  The pages join back to the exact original text (tested).
* **Comparison.** Each document becomes a vector with weight `(1 + ln tf) × idf'` per term (idf' is a smoothed idf that is never zero).
  Similarity is the **cosine**: dot product ÷ (length × length): 0 = nothing in common, 1 = identical. The page also lists shared terms,
  terms only in one document, and the overlap of authors and keywords (Jaccard). Real values: the federated-learning paper against the
  condition-monitoring report (same topic) = **0.226**; against the compression patent (different topic) = **0.031**.
* **Association analysis.**
  * *Related documents*: take the 25 heaviest terms of a document, collect candidates from their postings, score the candidates exactly.
  * *Similarity graph*: all pairs with cosine ≥ a threshold; the connected groups are the topics. At 0.10 on the sample data: 11 links,
    3 groups (wind-turbine monitoring, document databases, inverted-index retrieval).
  * *Keyword co-occurrence*: one MongoDB aggregation pipeline counts keyword pairs inside the database (`$project`, `$unwind` twice,
    keep each pair once, `$group`, `$sort`).
* **Download.** The file is streamed out of GridFS in 64 KB pieces into a folder, its SHA-256 is compared with the one stored at upload,
  an existing file is never overwritten (`name (1).ext`), and only the base name of the stored filename is used (no path tricks).
* **Delete.** In this order: the document's postings, then the GridFS file, then the record, then the counters. A half-finished delete
  therefore never leaves a document that can be found but not opened, and running it again finishes the job. Tests check that nothing
  is left in `documents`, `postings`, `fs.files` or `fs.chunks`.

### 2.6 The extras

* **Query language**: `"phrase"`, `-word`, `type:`, `author:`, `keyword:`, `title:`, `year:2022` / `2020..2023` / `>=2021`.
* **Facets**: counts by type, year, author and keyword over *all* matches; a click adds the filter to the query.
* **Did you mean**: an unknown word is replaced by the closest indexed word (edit distance 1, or 2 for long words; the commoner word wins).
* **Insights**: key terms (TF-IDF), a 3-sentence extractive summary (sentences scored by the weight of their words), reading time.
* **Analytics**: documents per year by type, top authors, co-authorship network.
* **Near-duplicates**: pairs above a cosine threshold; adding a document ≥ 0.6 similar to an existing one shows a warning.
* **Citations** (BibTeX, RIS, APA), **CSV/JSON export** (cells starting with `= + - @` are neutralised), **JSON API** (`/api`).

### 2.7 The website and the deployment

Flask serves the pages. For the public deployment the code is on GitHub, **Vercel** builds the Flask app (`app.py` is the entry point it
looks for), and the data is in **MongoDB Atlas**. The connection string is a Vercel environment variable, never in the code. Because a
public site has no login, it is **read-only by default**: the Add and Delete pages are removed and any write is refused with HTTP 403.
If the database is missing or unreachable the site shows a "Setup needed" page that says what is wrong.

---

## 3. Screenshots of the running system

| | |
|---|---|
| ![Documents](screenshots/01_home_documents.png) **Documents.** The home page lists every stored document with type, year, authors and file size. | ![Search](screenshots/02_search_bm25.png) **Ranked search.** BM25 results with matching words highlighted, a "why this score?" link, and facets on the right. |
| ![Query language](screenshots/03_advanced_query_facets.png) **Query language.** The page shows how the query was understood (terms, an excluded word, a year filter); the facets count all matches. | ![Spelling](screenshots/04_did_you_mean.png) **Did you mean.** A misspelled query gets a suggestion that is built from the words in the index. |
| ![Drawing](screenshots/05_document_drawing.png) **A drawing.** The binary PNG comes from GridFS; download and delete buttons, citations, related documents. | ![Insights](screenshots/06_document_paper_insights.png) **A paper with insights.** Automatic summary, key terms and reading time under the text. |
| ![Compare](screenshots/07_compare.png) **Comparison.** Cosine similarity bar, shared terms, terms only in one document, author and keyword overlap. | ![Graph](screenshots/08_similarity_graph.png) **Similarity graph.** Nodes are documents coloured by type; links are cosine ≥ 0.10; three topic groups. |
| ![Explain](screenshots/09_explain_score.png) **Why this score?** The BM25 score split into one line per query term. | ![Analytics](screenshots/10_analytics.png) **Analytics.** Documents per year by type, top authors, top keywords, co-authorship network. |
| ![Stats](screenshots/11_stats.png) **Statistics.** Documents by type, number of postings and terms, GridFS size, common keywords. | ![API](screenshots/12_api_docs.png) **JSON API.** The documented endpoints; every result of the website is also available as JSON. |
| ![Add](screenshots/13_add_document.png) **Add a document.** Metadata, text and any original file (not shown on the read-only public site). | |

Command-line output (real):

![CLI search](code/t01_cli_search.png)

![CLI compare](code/t02_cli_compare.png)

![CLI graph](code/t03_cli_graph.png)

![CLI co-occurrence](code/t04_cli_cooccur.png)

![Tests](code/t07_tests.png)

---

## 4. The code to explain

Open `docretrieval.py` next to the slides. Each block below is something you can read out loud in about 30 seconds. Line numbers are those of
the current file.

### 4.1 Text analysis (lines 91–113)

![tokenize and index_terms](code/c01_tokenize_index_terms.png)

*Say:* "`tokenize` lower-cases the text, splits it into words, drops stop words and one-letter words, and strips plural endings.
`index_terms` counts the result with weights: title words count three, keyword words two, everything else one. The same `tokenize` is applied
to a search query, which is why 'turbine' finds 'turbines'."
*Be ready to say:* only plurals are stripped, not a full stemmer.

### 4.2 Indexes (lines 348–354)

![ensure_indexes](code/c02_indexes.png)

*Say:* "These are created automatically when the program connects. The first one, on `postings` with `unique=True`, makes MongoDB itself refuse
a second record for the same term and document, and it also finds all documents of a term quickly. The others serve the filters."

### 4.3 Storing: file, record, index (lines 431–458)

![add_document](code/c03_add_document_write.png)

*Say:* "First the file goes to GridFS with `upload_from_stream`. Then the record goes into `documents`. Then all the postings go in with one
`insert_many`. Then the counters are updated with `$inc`. A single MongoDB server has no multi-document transactions, so the `except` block
undoes everything by hand: delete the postings, the record and the file, then re-raise the error. That gives all-or-nothing storage."

### 4.4 Search: fetch postings, then BM25 (lines 532–541 and 553–560)

![fetch postings](code/c04a_fetch_postings.png)

![BM25 scoring](code/c04b_bm25_scoring.png)

*Say:* "One query with `$in` fetches the postings of every query term, using the index. I remember each document's count for each term, and
count how many documents have each term (`df`). Then for every candidate and every matching term I add idf times the saturated term frequency
with length normalisation: that is BM25. Finally the list is sorted by score, ties broken by id so the order is stable."
*Point at line 558:* it is the formula on the ranking slide.

### 4.5 Comparison: TF-IDF and cosine (lines 670–681 and 200–206)

![tf-idf and compare](code/c05a_tfidf_compare.png)

![cosine](code/c05b_cosine.png)

*Say:* "`_tfidf` turns the counts of a document into weights: one plus the log of the count, times the idf. `compare` builds the two vectors,
finds the shared and distinctive terms by set operations, and the similarity is the cosine: the dot product divided by the two vector lengths."

### 4.6 Association: an aggregation pipeline (lines 783–796)

![aggregation pipeline](code/c06_aggregation_pipeline.png)

*Say:* "To count which keywords appear together I let MongoDB do it. `$project` copies the keywords array twice, `$unwind` turns each copy
into one document per keyword so every pair appears, the `$lt` match keeps each pair once, `$group` counts them, `$sort` and `$limit` finish.
The same function counts co-authors; only the field name changes."

### 4.7 Delete (lines 476–490)

![delete_document](code/c07_delete_document.png)

*Say:* "Postings first, so the document disappears from search immediately. Then the GridFS file; if it is already gone I ignore that, so a
repeated delete works. Then the record, then the counters with `$inc` of minus one."

![Delete output](code/t06_delete_cleanup.png)

### 4.8 Download (lines 909–922)

![download](code/c08_download_stream.png)

*Say:* "The file is read out of GridFS 64 KB at a time and written to disk while a SHA-256 is computed on the way. If the fingerprint differs from
the one stored at upload, I delete the partial file and refuse. If the GridFS file is missing, the user gets a clear error."

### 4.9 Deployment entry point (app.py, lines 29–37)

![app.py](code/c09_vercel_entry.png)

*Say:* "Vercel looks for `app`. This reads the connection string from an environment variable, and the site is read-only unless `READ_ONLY` is
set to 0. Indexes are created by whoever loads the data, so the public site does not need permission to create them."

### 4.10 Other places worth opening if asked

| Question | Where |
|---|---|
| How is the query parsed? | `parse_query` (about line 268) |
| How does "did you mean" work? | `suggest` and `_edit_distance` |
| How are facets counted? | the end of `search_ex` |
| How is the similarity graph built? | `similarity_graph`: one pass over the postings sorted by term, adding `w1 × w2` to each pair that shares the term |
| How do the web pages call the engine? | `webapp.py`, e.g. the `home()` route |
| How are things tested? | `test_docretrieval.py` (`test_bm25_matches_hand_computed_scores`, `test_delete_removes_document_postings_and_gridfs_file`) |

---

## 5. Slide-by-slide script

### Slide 1: Large-scale Document Retrieval System
*Introduction · in the 10-minute talk · about 0.5 min*

**On screen:** Title slide with the running search page on the right.

**Say:**

> Good day everyone. My topic is Experiment 5: a large-scale document retrieval system.
>
> It stores four kinds of documents - papers, patents, technical reports and engineering drawings - in MongoDB, and it lets you search them, read them, compare them, find related ones, and download the originals.
>
> Everything you will see is running for real on a MongoDB database. I will show the design first, then each function, and then the code behind it.
>
> (About half a minute here. The three numbers on the right are the whole story in short: four document types, six functions, 127 automatic tests.)

### Slide 2: What the system has to do
*Introduction · in the 10-minute talk · about 1 min*

**On screen:** Left: the four document types. Right: the six required functions. Bottom right: the technology.

**Say:**

> The assignment asks for a system that collects and stores four kinds of documents, and offers six functions: storage, retrieval, reading, comparison, association analysis and download. It has to be built on a NoSQL product.
>
> I chose MongoDB, with the pymongo driver, and GridFS for the files.
>
> One design point I want you to notice: the text documents and the engineering drawings go through the same system. A drawing is a real binary file - in my samples a PNG image - kept in GridFS. It can still be found, because it has a title, keywords and a description that are indexed like any other text.

### Slide 3: Three ways in, one engine, one database
*Design · in the 10-minute talk · about 1 min*

**On screen:** Architecture diagram: browser / other programs / terminal -> Flask site or command line -> DocumentStore engine -> MongoDB.

**Say:**

> This is the architecture. There are three ways in: the website in a browser, a JSON API for other programs, and a command line in the terminal.
>
> All three call the same engine - a Python class called DocumentStore, in the file docretrieval.py. The engine is the only part that talks to MongoDB, through the pymongo driver.
>
> That has a practical benefit: a feature is written once and works on the website, in the API and on the command line, and my automatic tests can test the engine directly.
>
> At the bottom is the database: the collections documents, postings and meta, plus GridFS for the original files. While developing I used a local MongoDB; the deployed website uses MongoDB Atlas, which is the same database in the cloud.

### Slide 4: Why MongoDB and GridFS
*Design · backup or 30-minute talk · about 1 min*

**On screen:** Three reasons for the technology choice, and the alternative that was rejected.

**Say:**

> Three reasons for this choice.
>
> One, flexible records. The four document types share one collection. A drawing has no body text and a patent has claims, but that does not need different tables. Authors and keywords are arrays inside the record, so reading one document is one read, with no join tables.
>
> Two, GridFS. A normal MongoDB record is limited to 16 megabytes and is not meant for big files. GridFS cuts a file into 255-kilobyte chunks stored in fs.chunks and describes the file in fs.files. So drawings and PDFs of any size live in the same database, are backed up with it, and are deleted with it.
>
> Three, the search index lives in the database too. The inverted index is just a collection with a unique index, so a search is an ordinary indexed query and there is nothing extra to install.
>
> I considered keeping files on disk and storing only their paths in the database. I rejected it: deleting a document could leave an orphan file behind, and the database and the files could be backed up at different moments.

### Slide 5: The data model (ER diagram)
*Design · in the 10-minute talk · about 2 min*

**On screen:** Entity-relationship diagram of the collections, GridFS and the embedded or derived concepts.

**Say:**

> This is the data model, as an entity-relationship diagram. There are five collections.
>
> In the middle, documents: one record per document, of any of the four types. It holds the title, type, year, abstract and text, and the authors and keywords as lists inside the record. Authors and keywords are drawn as dashed boxes because I embedded them instead of making separate collections: they are always read together with the document, which is the normal choice in a document database.
>
> On the right, postings is the inverted index. One record for each pair of a term and a document, with the weighted count of that term. The pair (term, doc_id) is the primary key, and I enforce it with a unique index. One document has many postings; a term is the key of many postings. The term box is dashed: there is no terms collection, the vocabulary is simply the set of values in postings.
>
> On the left is GridFS. fs.files has one record per stored file, and fs.chunks holds the bytes in pieces: one file has one or more chunks. A document has zero or one file, linked by file_id.
>
> Bottom right, meta holds two counters - the number of documents and their total length - which the ranking formula needs.
>
> Point at the cardinalities as you go: one document, zero-or-one file, zero-or-many postings.

### Slide 6: What one stored document looks like
*Design · in the 20-minute talk · about 1.5 min*

**On screen:** A real record from the database (pat-001, a patent).

**Say:**

> Here is a real record from my database, a patent with the id pat-001. The text is shortened for the slide.
>
> The _id is a readable id: the first letters show the type, here "pat" for patent. The type field is one of paper, patent, report or drawing.
>
> Authors and keywords are arrays inside the record.
>
> The field length is the number of weighted terms in the document - I will use it when I explain ranking.
>
> file_id points to the original file in GridFS, and next to it I keep the file name, type and size, plus a SHA-256 fingerprint. When someone downloads the file, I recompute the fingerprint and compare, so I can prove the file came back intact.
>
> A drawing uses exactly the same record. Its text is the description, and its file_id points to a binary PNG.

### Slide 7: The inverted index and the original file
*Design · backup or 30-minute talk · about 1 min*

**On screen:** Real postings records and meta counters (left); real fs.files and fs.chunks records (right).

**Say:**

> On the left, the inverted index. A posting says: this term occurs this many times in this document. For the patent pat-001, the word "posting" has a count of 7, "inverted" 4 and "compression" 2. The count is weighted: a word in the title counts three, in a keyword two, in the text one.
>
> My demo has ten documents, 945 postings and 701 distinct terms. The meta record shows the two counters: 10 documents and a total length of 1,617, which gives an average length of 161.7 terms.
>
> On the right, GridFS. fs.files describes the drawing: its name, its length of 1,016 bytes, the chunk size of 261,120 bytes - that is 255 kilobytes - and metadata that points back to the document. fs.chunks holds the bytes. The file is small, so it fits in one chunk, number 0, and you can see the PNG signature at the start of the data. A bigger file would simply have more chunks.

### Slide 8: Storing a document, step by step
*Design · in the 10-minute talk · about 1.5 min*

**On screen:** Six-step flow of add_document, and three guarantees underneath.

**Say:**

> This is what happens when I add a document. There are six steps.
>
> One, the input: the metadata, the text if I have it, and the original file.
>
> Two, validation: the title must not be empty, the type must be one of the four, the year must be plausible, and the id must not exist already. If this fails, nothing at all is written.
>
> Three, get the text: use the text I was given; otherwise read it from a .txt or .pdf file; for a drawing, the description is the text.
>
> Four, analyse the text: lower-case, split into words, drop stop words like "the" and "of", and strip plural endings, so "gearboxes" becomes "gearbox".
>
> Five, weight the terms: title words count three, keyword words two, everything else one.
>
> Six, write: first the file to GridFS, then the record to documents, then one posting per term, then update the counters.
>
> Three guarantees at the bottom. If any write fails, everything already written is removed again - all or nothing. A unique index makes MongoDB itself refuse a second posting for the same term and document. And the counters are updated with an atomic $inc, so ranking never has to scan the whole collection.

### Slide 9: Code: writing the file, the record and the index
*Design · backup or 30-minute talk · about 1.5 min*

**On screen:** docretrieval.py lines 431-458 (add_document), with numbered notes on the right.

**Say:**

> This is the heart of add_document, lines 431 to 458 of docretrieval.py.
>
> Inside the try block: first, if there is a file, it is uploaded to GridFS with upload_from_stream, and its name, type, size and SHA-256 are added to the record. Then the record goes into the documents collection with insert_one. Then all the postings go in with a single insert_many - one posting per term. Then the counters are updated with $inc.
>
> The except block is the safety net. MongoDB on a single server has no multi-document transactions, so I undo by hand: delete the postings, delete the record, and delete the GridFS file, then re-raise the error. That is how I get all-or-nothing storage.
>
> If asked: I tested this too - a test adds a document with a duplicate id and checks that nothing, no file and no posting, is left behind.

### Slide 10: Indexes keep every lookup fast
*Design · backup or 30-minute talk · about 1 min*

**On screen:** The index-creation code and what MongoDB reports for the two collections.

**Say:**

> Indexes are what make this scale. At the top is the code that creates them, run automatically when the program connects.
>
> The most important one is on postings: term and doc_id together, marked unique. It does two jobs. First, it finds all documents that contain a term very quickly, because term is the first part of the index. Second, it guarantees one record per term and document.
>
> There is a second index on doc_id alone, to rebuild a single document's term vector for comparison, and to find every posting of a document when I delete it.
>
> On documents, indexes on type and year, authors and keywords serve the filters and the keyword counts.
>
> The bottom left is the real output of MongoDB's getIndexes, which proves the unique index exists.

### Slide 11: From words to index terms
*Design · backup or 30-minute talk · about 1 min*

**On screen:** The tokenize and index_terms code, and a worked example.

**Say:**

> Before anything is indexed, text goes through a small analysis step. The function tokenize lower-cases the text, splits it on anything that is not a letter or digit, drops stop words and one-letter words, and strips plural endings. index_terms then counts the terms with weights: title three, keywords two, abstract and text one.
>
> The example on the right: the title "Gearboxes of wind turbines" becomes gearboxes, of, wind, turbines. The stop word "of" is dropped. Plurals are stripped, giving gearbox, wind, turbine. And because they come from the title, each gets weight three.
>
> The same function is applied to the search query, which is why a search for "turbine" finds a document that says "turbines".
>
> I should be honest about the limit: I only strip plurals, not a full stemmer, so "optimize" and "optimization" are different terms. That is on my list of next steps.

### Slide 12: Searching: the pipeline
*Functions · in the 20-minute talk · about 1 min*

**On screen:** Six-step search flow and three example queries in the query language.

**Say:**

> Now the first function, retrieval. This is what happens when you type a query.
>
> One, the query text. Two, it is parsed: words become terms, a phrase in quotes is remembered, a minus sign marks an excluded word, and words like type:patent or year:>2019 become filters. Three, one query on the postings collection fetches the records of all the query terms - this uses the unique index. Four, each candidate document is scored with BM25. Five, the filters are applied: type, year, author, keyword, title words, excluded words, and the exact phrase is checked in the stored text. Six, the results are ranked, with a snippet showing the matches, counts for the facets, and ten results per page. If nothing matches, I look for a spelling suggestion.
>
> At the bottom are three example queries. A phrase in quotes. A minus to exclude a word. And filters, which can be combined with words or used alone.
>
> It is all one function, search_ex, shared by the website, the API and the command line.

### Slide 13: Ranking with BM25
*Functions · in the 10-minute talk · about 2 min*

**On screen:** The BM25 formula in words, a worked example with real numbers, and the 'why this score?' page.

**Say:**

> Results are ranked with BM25, the standard ranking formula in search engines. The score of a document is the sum, over the query terms, of one contribution per term.
>
> Each contribution has three ideas. First, rare words weigh more: that is idf, the inverse document frequency. A word in only one document is strong evidence; a word in every document tells you nothing. Second, repeating a word helps, but with diminishing returns: the fifth occurrence adds much less than the first. Third, long documents are slightly penalised, so a long document does not win just by being long. The constants are k1 equals 1.5 and b equals 0.75, the usual values.
>
> A real example from my data: the patent pat-002 for the query "vibration sensor". "Vibration" appears 7 times in it and in 4 of the 10 documents, so its contribution is 1.832. "Sensor" appears 8 times and in 5 documents, so 1.454. The total is 3.286.
>
> On the right is the page I built to explain this: "why this score?". It shows each term, how often it occurs, how rare it is, and a bar for its contribution, and the numbers add up to exactly the score in the results list - I have a test that checks that.

### Slide 14: Code: fetching postings and scoring
*Functions · backup or 30-minute talk · about 1.5 min*

**On screen:** docretrieval.py lines 532-541 and 553-560, and the 'explain' command output.

**Say:**

> Here is the code. The top part, lines 532 to 541: one query on the postings collection with $in over all the query terms. For each posting I record the term count for that document, and I count how many documents have each term - that is df.
>
> The middle part, lines 553 to 560, is the BM25 formula exactly as on the previous slide: for each candidate document and each of its matching terms, idf times the saturated term frequency with length normalisation, added up. Then the list is sorted by score, and ties are broken by id so the order is stable.
>
> At the bottom is the same computation on the command line, with the explain command: the same two numbers, 1.832 and 1.454, adding up to 3.286.

### Slide 15: Query language, facets and spelling help
*Functions · in the 20-minute talk · about 1.5 min*

**On screen:** Two screenshots: an advanced query with facets, and a spelling suggestion.

**Say:**

> Let me show this on the website. On the left, the query "vibration sensor", minus "federated", with year greater than or equal to 2022. Under the search box the page shows how it understood the query: two terms, one excluded word, one year filter. Every result has a snippet with the matching words highlighted, and a "why this score" link.
>
> The sidebar on the right shows facets: how many results there are of each type, year, author and keyword. These count all the matches, not just this page, and clicking one adds that filter to the query.
>
> On the right-hand screenshot, I misspelled two words: "gearbxo vibraton". There are no results, but the page suggests "gearbox vibration". The suggestion uses the vocabulary of the index: for an unknown word it picks the closest indexed word, within one or two edits.
>
> Results come ten to a page, can be sorted by relevance or by newest first, and can be exported as CSV or JSON.

### Slide 16: Reading and downloading
*Functions · in the 20-minute talk · about 1.5 min*

**On screen:** A drawing's page with its image, and the download code.

**Say:**

> Reading and downloading. This is the page of a drawing. The picture is a real binary PNG, fetched from GridFS. The text under it is the description that is indexed.
>
> Reading is paginated. The text is cut into pages of about a thousand characters, preferably at a paragraph break and otherwise at a word break, so no word is split. The pages join back to exactly the original text - I test that.
>
> Downloading is the code on the right. It opens a stream from GridFS and copies the file out 64 kilobytes at a time, computing a SHA-256 on the way. If the fingerprint does not match the one stored when the file was added, the download is rejected and the partial file removed. It never overwrites an existing file; it adds a number to the name instead.
>
> On the website there is a security detail: images and PDFs may open in the browser, but any other file type, for example an uploaded HTML file, is forced to download, so an uploaded file can never run a script on my site.

### Slide 17: Comparing two documents
*Functions · in the 20-minute talk · about 1.5 min*

**On screen:** The compare page, and the TF-IDF and cosine code.

**Say:**

> Comparison. The page shows two documents: a similarity bar, the terms they share, the terms only one of them has, and how much their authors and keywords overlap.
>
> The similarity is TF-IDF cosine similarity. Each document becomes a vector: for each term, a weight equal to one plus the log of the count, times the idf. Cosine similarity is the dot product of the two vectors divided by the product of their lengths. Zero means nothing in common, one means identical.
>
> The code on the right: the first block builds the two weighted vectors from their postings and finds the shared and distinctive terms; the second block is the cosine itself, four lines.
>
> Real numbers: the federated-learning paper and the condition-monitoring report, which are both about wind-turbine gearboxes, score 0.226. The same paper against the compression patent, an unrelated topic, scores 0.031. The page also shows the shared author and the shared keywords - fault detection, wind turbine, vibration - with a Jaccard overlap.

### Slide 18: Association analysis
*Functions · in the 10-minute talk · about 2 min*

**On screen:** The similarity graph page, and the aggregation pipeline for keyword pairs.

**Say:**

> Association analysis has three parts.
>
> First, related documents: for one document, find the most similar ones. I take its 25 heaviest terms, collect candidate documents from the postings of those terms, and then score the candidates exactly with cosine similarity. That avoids comparing against every document.
>
> Second, the similarity graph, on the left. Every pair of documents with a cosine above a threshold is joined by a line. At a threshold of 0.10 on my ten documents there are 11 links and three clear topic groups: wind-turbine monitoring, document databases, and inverted-index retrieval. The groups are the connected components of the graph. Colours show the document type, so you can see a drawing, a report and a patent in the same group.
>
> Third, keyword co-occurrence, counting which keywords appear together in the same document. This one I let MongoDB do itself with an aggregation pipeline - the code on the right. It takes the keywords array twice, unwinds both copies, keeps each pair once, groups by pair and counts. For example "gearbox" with "wind turbine" appears together in three documents.

### Slide 19: Deleting without leaving traces
*Functions · in the 20-minute talk · about 1 min*

**On screen:** The delete code, the real output of a delete, and three points.

**Say:**

> Delete. The requirement was that deleting a document must also remove its index entries and its file. That is what this code does, in this order: first the postings, then the GridFS file, then the record, then the counters.
>
> The order is deliberate. If something fails half way, the document is already gone from search, because its postings were removed first, so there is never a ghost that can be found but cannot be opened. And running the delete again finishes the job.
>
> The output on the right is real. The demo adds a temporary document with a file, so the numbers go to 11 documents, 951 postings and 11 stored files. Then it deletes it: 6 postings and the file are removed, and the counts return to 10, 945 and 10. Zero postings still point at the deleted document.
>
> In the tests I check that nothing is left in documents, postings, fs.files or fs.chunks, and that search, the graph and related documents no longer see it.

### Slide 20: Beyond the requirements
*Beyond the requirements · backup or 30-minute talk · about 1 min*

**On screen:** Four extra pages: document insights, analytics, the JSON API and collection statistics.

**Say:**

> Beyond the six required functions I added a few things that make it feel like a real product.
>
> Document insights: on every document page, an automatic summary of three sentences chosen from the text, the key terms, and the reading time. The sentences are scored by the weight of their terms, so it uses the same index and needs no external AI service.
>
> Analytics: documents per year by type, the most active authors, and a co-authorship network: authors who share a document are joined by a line.
>
> A JSON API, so other programs can search and fetch documents, with the full query language.
>
> And a statistics page: counts by type, the size of the index, GridFS use, and the most common keywords.
>
> There are also citations in BibTeX, RIS and APA, CSV and JSON export, and a near-duplicate finder that warns you when you add a document that is almost the same as an existing one.

### Slide 21: The website and the cloud deployment
*Beyond the requirements · backup or 30-minute talk · about 1 min*

**On screen:** GitHub -> Vercel -> MongoDB Atlas, the Vercel entry point, and three safety points.

**Say:**

> The website is a small Flask application, and it is deployed. The code is on GitHub; Vercel builds the Flask app from it; and the data lives in MongoDB Atlas, the cloud version of the same database.
>
> app.py, shown here, is the entry point Vercel looks for. It reads the connection string from an environment variable. That variable is set in Vercel's settings, never written in the code, so the password is not on GitHub.
>
> Because a public site with no login is risky, it is read-only by default: the add and delete pages are removed and any write request is refused with a 403 error. Searching, reading, comparing and downloading all still work. Writes are enabled only with a setting, for a private deployment.
>
> If the database cannot be reached, or the setting is missing, the site shows a "Setup needed" page that says what is wrong instead of a blank error.

### Slide 22: Testing
*Beyond the requirements · in the 20-minute talk · about 1 min*

**On screen:** The real pytest output, and what the tests cover.

**Say:**

> Testing. I wrote 127 automatic tests and they all pass, in about 40 seconds. This is the real output.
>
> They run against a real MongoDB, not mocks: each test run creates a throw-away database and drops it afterwards.
>
> They cover each required function: ranking, including a hand-computed BM25 score; the filters; comparison; association analysis, checked against manual counts; a download that must be byte-for-byte identical; and delete cleanup. They also cover the query language, every web page, and the JSON API.
>
> One more check I am proud of: I broke the code on purpose - I removed the posting deletion, removed the length normalisation in BM25, disabled the phrase check, the exclusion and the spelling distance - and each time the matching test failed. That shows the tests really can fail.

### Slide 23: Limits and next steps
*Wrap-up · in the 20-minute talk · about 1 min*

**On screen:** Honest limits on the left, ideas for improvement on the right.

**Say:**

> Honest limits. I only strip plural endings, so word forms like "optimize" and "optimization" are different terms. The similarity graph compares all pairs, so it suits thousands of documents, not millions; for a very large collection you would use the related-documents search instead. A plain MongoDB server has no multi-document transactions, so I undo failed writes by hand. Text is extracted only from text files and PDFs: scanned PDFs and drawings are searched through their title, keywords and description. And the public website has no login, which is why it is read-only.
>
> Next steps: a proper stemmer and a synonym list; MongoDB Atlas Search or vector search for meaning-based matching; a login with an admin role so writes can be allowed safely; transactions on a replica set; and OCR to read scanned documents.

### Slide 24: Thank you: questions?
*Wrap-up · in the 10-minute talk · about 0.5 min*

**On screen:** Closing slide with the live demo plan.

**Say:**

> That is the system: four document types, six functions, built on MongoDB and GridFS, with the data model, the index and the code that I showed.
>
> If there is time, I can do a short live demo. The plan is on the right: search for a phrase, show the spelling suggestion, open a drawing and download it, compare two documents, show the similarity graph, and explain a score.
>
> Thank you. I am happy to take questions.
>
> (Before the demo: start MongoDB, run "python webapp.py --db docretrieval_demo", open http://127.0.0.1:5000 . Keep a browser tab on each of the pages in advance.)


---

## 6. Timing

| Talk length | Slides | Notes |
|---|---|---|
| **10 minutes** | 1, 2, 3, 5, 8, 13, 18, 24 | Design, storing, ranking, association. Skip the code slides. |
| **20 minutes** | the 10-minute set plus 6, 12, 15, 16, 17, 19, 22, 23 | Adds the individual functions and testing. |
| **30 minutes** | all 24 | Adds the code slides and the extras. |

Keep slides 4, 7, 9, 10, 11, 14, 20, 21 as **backup**: skip them unless someone asks.

---

## 7. Live demo

**Before you present**

1. Start MongoDB (installed locally, or use Atlas). See `README.md` section 2.
2. In the project folder: `python docretrieval.py demo` (loads the 10 sample documents into the database `docretrieval_demo`).
3. `python webapp.py --db docretrieval_demo`, then open <http://127.0.0.1:5000> and keep these tabs open:
   `/`, `/?q="posting list"`, `/?q=gearbxo vibraton`, `/doc/drw-001`, `/compare?a=pap-003&b=rep-002`, `/graph?threshold=0.10`,
   `/explain/pat-002?q=vibration sensor`.
4. Have a terminal open in the project folder, and `docretrieval.py` open in your editor at the lines in section 4.

**The demo (about 5 minutes)**

| Step | Do | What it shows |
|---|---|---|
| 1 | Search `inverted index compression` | BM25 ranking, highlighted snippets: `pat-001` first (7.436) |
| 2 | Search `"posting list"` | exact phrase: only `pat-001` |
| 3 | Search `vibration -federated type:report` | the query language: one excluded word, a filter |
| 4 | Search `gearbxo vibraton` | the spelling suggestion |
| 5 | Open drawing `drw-001`, click **Download original** | binary file from GridFS, same bytes as stored |
| 6 | Compare `pap-003` and `rep-002` | cosine 0.226, shared terms, shared author |
| 7 | Open the similarity graph | three topic groups |
| 8 | Click "why this score?" on a result | the BM25 table: 1.832 + 1.454 = 3.286 |
| 9 | Terminal: `python docretrieval.py demo`, then point at step 8 of its output | a document is added, then deleted: the counts return to 10 / 945 / 10 |

If the network or database fails, the screenshots in section 3 and the real terminal output in `code/` show the same things.

---

## 8. Questions you may get

**Why MongoDB and not a relational database?** The four document types differ (a drawing has no body text, a patent has claims) and authors/keywords
are lists, so one flexible record per document fits better than several tables with joins. GridFS stores big binary files in the same database.
(A relational system could do it too; the assignment asks for NoSQL.)

**What is GridFS, and why not put the file inside the document?** A MongoDB record is limited to 16 MB and is not meant for large binary
data. GridFS cuts a file into 255 KB chunks (`fs.chunks`) described by one `fs.files` record, with no size limit in practice, and it is part of the
same database, so backups and deletes cover it.

**Why not use MongoDB's built-in text index?** I built my own inverted index to implement and show the ranking: BM25 with my own weights (title ×3,
keyword ×2), phrase and exclusion handling, facets and score explanation. MongoDB's `$text` has its own scoring and less control. In production,
Atlas Search would be the natural next step.

**What is the difference between BM25 and TF-IDF here?** Both reward rare words that occur often. BM25 adds saturation (the 10th occurrence adds
little) and document-length normalisation, so it is better for *ranking results for a query*. I use TF-IDF vectors with cosine similarity for
*comparing two documents*, where I need a vector for each document.

**How do you keep the index consistent with the documents?** Adding writes the file, the record and the postings in that order and undoes everything
on an error. Deleting removes postings first. The corpus counters use atomic `$inc`, and `rebuild_corpus_stats()` can recompute them from the
documents if they ever drift. A unique index on `(term, doc_id)` stops duplicate postings.

**What if two users add documents at the same time?** Each document has its own id (the `_id` index rejects a duplicate id), postings are unique per
`(term, doc_id)`, and the counters are updated atomically with `$inc`, so concurrent adds do not corrupt the index.

**Does it scale?** A search is one indexed lookup per query term, so it grows with the number of matching postings, not with the whole collection.
The similarity graph compares all pairs, which is fine for thousands of documents but not millions; for big collections use "related documents",
which only scores candidates. The data could be sharded (for example `postings` by term); I did not implement that.

**How are drawings searched?** Through their title, keywords and description, which are indexed like any text. The file itself (PNG) is stored as
bytes. Reading text out of a drawing would need OCR or CAD metadata: a next step.

**What about PDFs?** If `pypdf` is installed, text is extracted when a PDF is added. Scanned PDFs have no text layer, so they would need OCR.

**How do you know the ranking is right?** A test recomputes the BM25 score from the stored data and compares it with the engine's score. A second test
checks that the "why this score?" breakdown adds up to the same number. I also broke the code on purpose and saw the tests fail.

**Is it secure?** The public site is read-only (no login), stored text is HTML-escaped, uploaded files other than images and PDFs are forced to
download so they cannot run scripts, stored filenames are reduced to a base name, CSV exports neutralise formula cells, and the database password is
only in Vercel's environment settings. There is **no authentication**: the next step would be a login with an admin role.

**How does "did you mean" work?** For a query word that occurs nowhere in the index, I look for the indexed word with the smallest edit distance
(1 edit for short words, 2 for longer ones) and prefer the one found in more documents.

**What is the difference between "related documents" and the "similarity graph"?** Related documents answers "what is like *this* document?" and only
scores candidates sharing its heaviest terms. The graph answers "how do *all* documents group?" by comparing every pair above a threshold.

**Why embed authors and keywords?** They are always read with the document, so embedding gives one read and no joins. The cost is that renaming an
author would need an update in every document, which this system never does. Counting co-authors or keyword pairs is done with an aggregation pipeline.

**Why `k1 = 1.5` and `b = 0.75`?** They are the common defaults for BM25: `k1` controls how fast repeated words stop helping, `b` how strongly length is
normalised (0 = ignore length, 1 = full normalisation).

**What changes on a replica set?** You could use multi-document transactions for add and delete, instead of the manual undo.

**How do you back it up?** `mongodump` of the database includes the GridFS collections, so documents, index and files are backed up together: one of the reasons not to keep files on disk.

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **Inverted index** | For every word, the list of documents that contain it; the reverse of "document → words" |
| **Posting** | One entry in that list: *this word, in this document, this many times* (here one record per term and document) |
| **tf** (term frequency) | How often a word occurs in a document (here weighted: title ×3, keyword ×2, text ×1) |
| **df** (document frequency) | In how many documents a word occurs |
| **idf** (inverse document frequency) | A weight that is high for rare words and low for common ones |
| **TF-IDF** | tf combined with idf: how characteristic a word is of a document |
| **BM25** | The standard ranking formula: idf, saturating tf, and length normalisation |
| **Cosine similarity** | How similar two word-weight vectors are (the cosine of the angle): 0 to 1 |
| **Jaccard overlap** | Shared items ÷ all items of two sets (used for authors and keywords) |
| **Facet** | A count shown beside the results (how many are patents, from 2022, by one author...) that filters when clicked |
| **GridFS** | MongoDB's way to store big files: cut into chunks, described by a metadata record |
| **Chunk** | One piece (here up to 255 KB) of a GridFS file |
| **Aggregation pipeline** | A chain of MongoDB stages (`$match`, `$unwind`, `$group`, `$sort`...) that computes results inside the database |
| **Embedded document / array** | Data stored inside a record instead of in another collection |
| **Unique index** | An index that refuses two records with the same key |
| **Atomic update** | An update (like `$inc`) that cannot be half done or lost when two happen at once |
| **Read-only mode** | A setting that removes every write feature, used for the public website |

---

## 10. Regenerating the material

```bash
python docretrieval.py demo           # loads the sample data
python docs/make_images.py            # code, data and terminal images in docs/code/
python webapp.py --db docretrieval_demo   # the site the screenshots come from
```

The ER, architecture and flow diagrams are in `docs/diagrams/diagrams_large.html` and `docs/diagrams/diagrams.html` (open them through any
local web server and screenshot the sections).
