#!/usr/bin/env python3
"""Large-scale document retrieval system on MongoDB (pymongo + GridFS).

Experiment 5: store papers, patents, technical reports and engineering
drawings; search them with BM25 over an inverted index kept in MongoDB;
read, compare and analyse associations between documents; download originals.

Collections
    documents  one record per document: metadata + extracted text
    postings   inverted index, one record per (term, doc_id), unique index
    meta       a single "corpus" record: document count and total length
    fs.files / fs.chunks   GridFS, holds the original files (PDF, TXT, PNG, ...)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from itertools import groupby

import gridfs
from bson import ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

DEFAULT_URI = "mongodb://localhost:27017"
DEFAULT_DB = "docretrieval"
DEMO_DB = "docretrieval_demo"

DOC_TYPES = ("paper", "patent", "report", "drawing")
TYPE_ALIASES = {
    "technical_report": "report", "technical-report": "report",
    "technical report": "report", "techreport": "report", "tr": "report",
    "engineering_drawing": "drawing", "engineering drawing": "drawing",
}
TYPE_PREFIX = {"paper": "pap", "patent": "pat", "report": "rep", "drawing": "drw"}

# BM25 parameters and field boosts used when building the index.
K1 = 1.5
B = 0.75
TITLE_BOOST = 3
KEYWORD_BOOST = 2

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tex", ".rst", ".log"}

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
he her his how if in into is it its may more most no not of on one or other our
over only every show shows she should so some such than that the their them then there these they this
those to under up use used using was we were what when which while who will with
would you your also each both any all about after before between during via per
""".split())


class DocError(Exception):
    """Base class for errors reported to the user."""


class NotFound(DocError):
    pass


# --------------------------------------------------------------------------
# Text analysis
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Very light plural stripping so 'networks' matches 'network'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stop words, stem."""
    out = []
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


def index_terms(title: str, keywords: list[str], abstract: str, text: str) -> Counter:
    """Weighted term frequencies: title x3, keywords x2, abstract/body x1."""
    counts: Counter = Counter()
    for t in tokenize(title):
        counts[t] += TITLE_BOOST
    for kw in keywords:
        for t in tokenize(kw):
            counts[t] += KEYWORD_BOOST
    for t in tokenize(abstract):
        counts[t] += 1
    for t in tokenize(text):
        counts[t] += 1
    return counts


def make_snippet(text: str, query_terms: set[str], width: int = 28) -> tuple[str, int]:
    """Best window of `width` words, matches wrapped in **. Returns (snippet, hits)."""
    words = list(re.finditer(r"\S+", text or ""))
    if not words:
        return "", 0
    hit = [any(t in query_terms for t in tokenize(w.group())) for w in words]
    prefix = [0]
    for h in hit:
        prefix.append(prefix[-1] + h)
    width = min(width, len(words))
    best_start, best_hits = 0, -1
    for s in range(len(words) - width + 1):
        n = prefix[s + width] - prefix[s]
        if n > best_hits:
            best_start, best_hits = s, n
    first = next((i for i in range(best_start, best_start + width) if hit[i]), None)
    start = best_start if first is None else max(best_start, first - 6)
    end = min(len(words), start + width)
    parts = []
    for i in range(start, end):
        w = words[i].group()
        if hit[i]:
            lead, core, trail = re.match(r"^(\W*)(.*?)(\W*)$", w, re.S).groups()
            w = f"{lead}**{core}**{trail}"
        parts.append(w)
    snippet = " ".join(parts)
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(words) else ""), max(best_hits, 0)


def split_pages(text: str, size: int) -> list[str]:
    """Cut text into pages of about `size` characters, preferring paragraph then
    word boundaries. Pages concatenate back to the exact original text."""
    pages, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        if j < n:
            lo = i + size * 3 // 4
            k = text.rfind("\n\n", lo, j)
            if k != -1:
                j = k + 2
            else:
                k = max(text.rfind(" ", lo, j), text.rfind("\n", lo, j))
                if k != -1:
                    j = k + 1
        pages.append(text[i:j])
        i = j
    return pages or [""]


def extract_text(filename: str, data: bytes) -> str:
    """Best-effort text extraction for plain-text files and PDFs."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            import pypdf  # optional dependency
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    return ""


def _norm_type(doc_type: str) -> str:
    t = (doc_type or "").strip().lower()
    t = TYPE_ALIASES.get(t, t)
    if t not in DOC_TYPES:
        raise DocError(f"unknown document type {doc_type!r}; expected one of {', '.join(DOC_TYPES)}")
    return t


def _dedupe(values, lower=False) -> list[str]:
    seen, out = set(), []
    for v in values or []:
        v = " ".join(str(v).split())
        if lower:
            v = v.lower()
        if v and v.casefold() not in seen:
            seen.add(v.casefold())
            out.append(v)
    return out


def _cosine(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b[t] for t, w in a.items() if t in b)
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {x.casefold() for x in a}, {x.casefold() for x in b}
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _safe_filename(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/")) or "download.bin"
    return re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)


# --------------------------------------------------------------------------
# Query language:  "exact phrase"  -excluded  type:patent  author:berg
#                  keyword:"wind turbine"  title:gearbox  year:2020..2023
# --------------------------------------------------------------------------
_QUERY_TOKEN = re.compile(r'(-?)(?:([A-Za-z]+):)?("([^"]*)"|\S+)')
QUERY_FIELDS = ("type", "author", "keyword", "title", "year")


def _norm_words(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class ParsedQuery:
    terms: list = dc_field(default_factory=list)        # stemmed ranking terms
    phrases: list = dc_field(default_factory=list)      # normalised phrases, must appear verbatim
    exclude: list = dc_field(default_factory=list)      # stemmed terms that must not appear
    doc_type: str | None = None
    authors: list = dc_field(default_factory=list)
    keywords: list = dc_field(default_factory=list)
    title_terms: list = dc_field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None

    def describe(self) -> dict:
        filters = []
        if self.doc_type:
            filters.append(f"type: {self.doc_type}")
        filters += [f"author: {a}" for a in self.authors]
        filters += [f"keyword: {k}" for k in self.keywords]
        if self.title_terms:
            filters.append("title has: " + " ".join(self.title_terms))
        if self.year_from is not None or self.year_to is not None:
            filters.append(f"year: {self.year_from if self.year_from is not None else ''}..{self.year_to if self.year_to is not None else ''}")
        return {"terms": self.terms, "phrases": self.phrases, "exclude": self.exclude, "filters": filters}


def _parse_year(value: str) -> tuple[int | None, int | None]:
    v = value.strip()
    m = re.fullmatch(r"(\d{4})?\.\.(\d{4})?", v)
    if m and (m.group(1) or m.group(2)):
        return (int(m.group(1)) if m.group(1) else None), (int(m.group(2)) if m.group(2) else None)
    m = re.fullmatch(r"(>=|<=|>|<)?(\d{4})", v)
    if not m:
        raise DocError(f"cannot read year filter {value!r}; use year:2022, year:2020..2023, year:>2020 or year:<=2023")
    op, y = m.group(1), int(m.group(2))
    return {None: (y, y), ">=": (y, None), ">": (y + 1, None), "<=": (None, y), "<": (None, y - 1)}[op]


def parse_query(q: str) -> ParsedQuery:
    pq = ParsedQuery()
    for m in _QUERY_TOKEN.finditer(q or ""):
        neg, fld, body, quoted = m.group(1), (m.group(2) or "").lower(), m.group(3), m.group(4)
        value = quoted if quoted is not None else body
        if fld and fld not in QUERY_FIELDS:       # e.g. "http://..." is just text
            fld, neg, value, quoted = "", "", m.group(0).lstrip("-"), None
        if fld:
            if neg:
                raise DocError("'-' can only be put in front of a plain word, for example -turbine")
            value = value.strip()
            if not value:
                continue
            if fld == "type":
                pq.doc_type = _norm_type(value)
            elif fld == "author":
                pq.authors.append(value)
            elif fld == "keyword":
                pq.keywords.append(value)
            elif fld == "title":
                pq.title_terms += tokenize(value)
            else:
                lo, hi = _parse_year(value)
                if lo is not None:
                    pq.year_from = lo if pq.year_from is None else max(pq.year_from, lo)
                if hi is not None:
                    pq.year_to = hi if pq.year_to is None else min(pq.year_to, hi)
            continue
        if neg:
            pq.exclude += tokenize(value)
        elif quoted is not None:
            words = _norm_words(value)
            if words:
                pq.phrases.append(" ".join(words))
                pq.terms += tokenize(value)
        else:
            pq.terms += tokenize(value)
    pq.terms = list(dict.fromkeys(pq.terms))
    pq.exclude = list(dict.fromkeys(pq.exclude))
    pq.title_terms = list(dict.fromkeys(pq.title_terms))
    return pq


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, giving up (returning limit + 1) once it must exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _bibtex_escape(text: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", text or "")


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class DocumentStore:
    def __init__(self, uri: str = DEFAULT_URI, db_name: str = DEFAULT_DB, client: MongoClient | None = None,
                 create_indexes: bool = True):
        self.client = client or MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.client.admin.command("ping")  # fail fast if no server
        self.db = self.client[db_name]
        self.documents = self.db["documents"]
        self.postings = self.db["postings"]
        self.meta = self.db["meta"]
        self.fs = gridfs.GridFSBucket(self.db)
        self._vocab: dict[str, int] | None = None
        self._vocab_at = 0.0
        if create_indexes:
            self.ensure_indexes()

    def ensure_indexes(self) -> None:
        self.postings.create_index([("term", ASCENDING), ("doc_id", ASCENDING)], unique=True, name="term_doc_unique")
        self.postings.create_index([("doc_id", ASCENDING)], name="doc_id")
        self.documents.create_index([("type", ASCENDING), ("year", ASCENDING)], name="type_year")
        self.documents.create_index([("authors", ASCENDING)], name="authors")
        self.documents.create_index([("keywords", ASCENDING)], name="keywords")
        self.documents.create_index([("year", ASCENDING)], name="year")

    def drop_database(self) -> None:
        self.client.drop_database(self.db.name)

    def close(self) -> None:
        self.client.close()

    # ---- corpus statistics ------------------------------------------------
    def _corpus(self) -> tuple[int, int]:
        m = self.meta.find_one({"_id": "corpus"}) or {}
        return int(m.get("n", 0)), int(m.get("total_len", 0))

    def rebuild_corpus_stats(self) -> None:
        """Recompute the counters from the documents collection (repair tool)."""
        agg = list(self.documents.aggregate([{"$group": {"_id": None, "n": {"$sum": 1}, "t": {"$sum": "$length"}}}]))
        n, t = (agg[0]["n"], agg[0]["t"]) if agg else (0, 0)
        self.meta.update_one({"_id": "corpus"}, {"$set": {"n": n, "total_len": t}}, upsert=True)

    def _df(self, terms) -> dict[str, int]:
        terms = list(terms)
        if not terms:
            return {}
        cur = self.postings.aggregate([
            {"$match": {"term": {"$in": terms}}},
            {"$group": {"_id": "$term", "df": {"$sum": 1}}},
        ])
        return {r["_id"]: r["df"] for r in cur}

    @staticmethod
    def _idf_smooth(n: int, df: int) -> float:
        return math.log((n + 1) / (df + 1)) + 1.0

    # ---- storage ----------------------------------------------------------
    def add_document(self, title: str, doc_type: str, authors=None, keywords=None, year: int | None = None,
                     abstract: str = "", text: str = "", file_path: str | None = None,
                     file_bytes: bytes | None = None, filename: str | None = None,
                     doc_id: str | None = None) -> dict:
        """Store a document: metadata + text in `documents`, the original file in
        GridFS, and its terms in `postings`. Nothing is written if validation fails;
        a failure part-way through is rolled back."""
        title = " ".join((title or "").split())
        if not title:
            raise DocError("title is required")
        doc_type = _norm_type(doc_type)
        if year is not None:
            year = int(year)
            if not 1000 <= year <= 2200:
                raise DocError(f"implausible year {year}")
        if file_path and file_bytes is not None:
            raise DocError("give either file_path or file_bytes, not both")
        if file_path and not os.path.isfile(file_path):
            raise DocError(f"file not found: {file_path}")
        if file_path:
            with open(file_path, "rb") as fh:
                file_bytes = fh.read()
            filename = filename or os.path.basename(file_path)
        doc_id = doc_id or f"{TYPE_PREFIX[doc_type]}-{uuid.uuid4().hex[:8]}"
        if self.documents.count_documents({"_id": doc_id}, limit=1):
            raise DocError(f"document id already exists: {doc_id}")

        authors = _dedupe(authors)
        keywords = _dedupe(keywords, lower=True)
        abstract = (abstract or "").strip()
        text = (text or "").strip()
        if not text and file_bytes is not None:
            text = extract_text(filename or "", file_bytes).strip()
        if not text:
            text = abstract
        terms = index_terms(title, keywords, abstract, text)

        record = {
            "_id": doc_id, "title": title, "type": doc_type, "authors": authors, "keywords": keywords,
            "year": year, "abstract": abstract, "text": text, "length": sum(terms.values()),
            "file_id": None, "filename": None, "content_type": None, "size": None, "sha256": None,
            "added_at": datetime.now(timezone.utc),
        }
        file_id = None
        try:
            if file_bytes is not None:
                filename = _safe_filename(filename or f"{doc_id}.bin")
                ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                file_id = self.fs.upload_from_stream(
                    filename, io.BytesIO(file_bytes),
                    metadata={"doc_id": doc_id, "content_type": ctype, "doc_type": doc_type})
                record.update(file_id=file_id, filename=filename, content_type=ctype, size=len(file_bytes),
                              sha256=hashlib.sha256(file_bytes).hexdigest())
            self.documents.insert_one(record)
            if terms:
                self.postings.insert_many(
                    [{"term": t, "doc_id": doc_id, "tf": tf} for t, tf in terms.items()], ordered=False)
            self.meta.update_one({"_id": "corpus"}, {"$inc": {"n": 1, "total_len": record["length"]}}, upsert=True)
        except Exception:
            # No multi-document transactions on a standalone server: undo by hand.
            self.postings.delete_many({"doc_id": doc_id})
            self.documents.delete_one({"_id": doc_id})
            if file_id is not None:
                try:
                    self.fs.delete(file_id)
                except gridfs.errors.NoFile:
                    pass
            raise
        self._vocab = None
        return {"doc_id": doc_id, "type": doc_type, "terms": len(terms), "length": record["length"],
                "file_id": file_id, "size": record["size"]}

    def get_document(self, doc_id: str, with_text: bool = True) -> dict:
        proj = None if with_text else {"text": 0}
        doc = self.documents.find_one({"_id": doc_id}, proj)
        if not doc:
            raise NotFound(f"no such document: {doc_id}")
        return doc

    def list_documents(self, doc_type: str | None = None, limit: int = 100) -> list[dict]:
        q = {"type": _norm_type(doc_type)} if doc_type else {}
        cur = self.documents.find(q, {"text": 0, "abstract": 0}).sort([("type", 1), ("year", 1), ("_id", 1)]).limit(limit)
        return list(cur)

    def delete_document(self, doc_id: str) -> dict:
        """Remove the document, every posting that points at it, and its GridFS file.
        Postings go first so a half-finished delete never leaves a searchable ghost;
        re-running the delete finishes the job."""
        doc = self.documents.find_one({"_id": doc_id}, {"text": 0})
        if not doc:
            raise NotFound(f"no such document: {doc_id}")
        removed = self.postings.delete_many({"doc_id": doc_id}).deleted_count
        file_removed = False
        if doc.get("file_id") is not None:
            try:
                self.fs.delete(doc["file_id"])
                file_removed = True
            except gridfs.errors.NoFile:
                pass
        self.documents.delete_one({"_id": doc_id})
        self.meta.update_one({"_id": "corpus"}, {"$inc": {"n": -1, "total_len": -doc["length"]}})
        self._vocab = None
        return {"doc_id": doc_id, "postings_removed": removed, "file_removed": file_removed}

    # ---- retrieval --------------------------------------------------------
    def search(self, query: str, doc_type: str | None = None, year_from: int | None = None,
               year_to: int | None = None, author: str | None = None, limit: int = 10,
               require_all: bool = False, snippet: bool = True) -> list[dict]:
        """BM25-ranked search; returns the best `limit` results (see search_ex for the rest)."""
        return self.search_ex(query, doc_type, year_from, year_to, author, limit=limit,
                              require_all=require_all, snippet=snippet)["results"]

    def search_ex(self, query: str, doc_type: str | None = None, year_from: int | None = None,
                  year_to: int | None = None, author: str | None = None, limit: int = 10, offset: int = 0,
                  require_all: bool = False, snippet: bool = True, sort: str = "relevance") -> dict:
        """Full search: results page, total match count, facets, spelling suggestion.

        The query may use "phrases", -exclusions and type:/author:/keyword:/title:/year: filters
        (see parse_query). The keyword arguments are extra filters that are ANDed with those.
        Any query term may match (all of them with require_all). sort: relevance | year."""
        pq = parse_query(query)
        types = {t for t in (pq.doc_type, _norm_type(doc_type) if doc_type else None) if t}
        los = [y for y in (pq.year_from, int(year_from) if year_from is not None else None) if y is not None]
        his = [y for y in (pq.year_to, int(year_to) if year_to is not None else None) if y is not None]
        lo, hi = (max(los) if los else None), (min(his) if his else None)
        authors = list(pq.authors) + ([author.strip()] if author and author.strip() else [])
        out = {"results": [], "total": 0, "query": pq.describe(), "suggestion": None,
               "facets": {"type": [], "year": [], "author": [], "keyword": []}}
        has_filters = bool(types or lo is not None or hi is not None or authors or pq.keywords or pq.title_terms)
        n, total_len = self._corpus()
        if n == 0 or len(types) > 1 or not (pq.terms or has_filters):
            return out
        avgdl = total_len / n

        flt: dict = {}
        if types:
            flt["type"] = next(iter(types))
        if lo is not None or hi is not None:
            flt["year"] = {**({"$gte": lo} if lo is not None else {}), **({"$lte": hi} if hi is not None else {})}
        conds = [{"authors": {"$elemMatch": {"$regex": re.escape(a), "$options": "i"}}} for a in authors]
        conds += [{"keywords": {"$elemMatch": {"$regex": re.escape(k), "$options": "i"}}} for k in pq.keywords]
        if conds:
            flt["$and"] = conds

        by_doc: dict[str, dict[str, int]] = defaultdict(dict)
        df: Counter = Counter()
        if pq.terms:
            for p in self.postings.find({"term": {"$in": pq.terms}}, {"_id": 0}):
                by_doc[p["doc_id"]][p["term"]] = p["tf"]
                df[p["term"]] += 1
            if require_all:
                by_doc = {d: m for d, m in by_doc.items() if len(m) == len(pq.terms)}
            if not by_doc:
                out["suggestion"] = self.suggest(query)
                return out
            flt["_id"] = {"$in": list(by_doc)}
        cur = self.documents.find(flt, {"title": 1, "type": 1, "year": 1, "authors": 1, "keywords": 1, "length": 1})
        meta = {d["_id"]: d for d in (cur if pq.terms else cur.limit(5000))}
        if pq.exclude:
            for gone in self.postings.distinct("doc_id", {"term": {"$in": pq.exclude}}):
                meta.pop(gone, None)
        if pq.title_terms:
            need = set(pq.title_terms)
            meta = {i: d for i, d in meta.items() if need <= set(tokenize(d["title"]))}

        scored = []
        for doc_id, d in meta.items():
            score, dl = 0.0, d["length"] or 1
            for t, tf in by_doc.get(doc_id, {}).items():
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
            scored.append((score, doc_id))
        scored.sort(key=lambda x: (-x[0], x[1]))

        if pq.phrases and scored:  # phrases are verified against the stored text of the best candidates
            scored = scored[:2000]
            full = {d["_id"]: d for d in self.documents.find(
                {"_id": {"$in": [i for _, i in scored]}}, {"title": 1, "abstract": 1, "text": 1})}

            def has_all_phrases(doc_id):
                d = full[doc_id]
                hay = " " + " ".join(_norm_words(f"{d['title']} {d.get('abstract', '')} {d.get('text', '')}")) + " "
                return all(f" {ph} " in hay for ph in pq.phrases)
            scored = [(sc, i) for sc, i in scored if has_all_phrases(i)]
        if sort == "year":
            scored.sort(key=lambda x: (-(meta[x[1]]["year"] or 0), -x[0], x[1]))

        out["total"] = len(scored)
        facet_src = [meta[i] for _, i in scored]
        out["facets"] = {
            "type": Counter(d["type"] for d in facet_src).most_common(),
            "year": sorted(Counter(d["year"] for d in facet_src if d["year"]).items(), reverse=True),
            "author": Counter(a for d in facet_src for a in d["authors"]).most_common(8),
            "keyword": Counter(k for d in facet_src for k in d["keywords"]).most_common(8),
        }
        page = scored[max(offset, 0):max(offset, 0) + limit]
        texts = {}
        if snippet and page:
            texts = {d["_id"]: d for d in self.documents.find({"_id": {"$in": [i for _, i in page]}}, {"text": 1, "abstract": 1})}
        qset = set(pq.terms)
        for score, doc_id in page:
            d = meta[doc_id]
            r = {"doc_id": doc_id, "title": d["title"], "type": d["type"], "year": d["year"],
                 "authors": d["authors"], "score": round(score, 4), "matched_terms": sorted(by_doc.get(doc_id, {}))}
            if snippet:
                src = texts.get(doc_id, {})
                snip, hits = make_snippet(src.get("text", ""), qset)
                if hits == 0 and src.get("abstract"):
                    snip = make_snippet(src["abstract"], qset)[0]
                r["snippet"] = snip
            out["results"].append(r)
        if pq.terms and (not scored or any(df[t] == 0 for t in pq.terms)):
            out["suggestion"] = self.suggest(query)
        return out

    def _vocabulary(self, ttl: float = 60.0) -> dict[str, int]:
        """term -> document frequency, cached for `ttl` seconds (used for spelling suggestions)."""
        if self._vocab is None or time.monotonic() - self._vocab_at > ttl:
            self._vocab = {r["_id"]: r["df"] for r in self.postings.aggregate(
                [{"$group": {"_id": "$term", "df": {"$sum": 1}}}], allowDiskUse=True)}
            self._vocab_at = time.monotonic()
        return self._vocab

    def suggest(self, query: str) -> str | None:
        """'Did you mean': replace plain query words that occur nowhere in the index by the
        closest indexed term (edit distance 1, or 2 for longer words; the commoner term wins)."""
        vocab = self._vocabulary()
        pieces, last, changed = [], 0, False
        for m in _QUERY_TOKEN.finditer(query or ""):
            if m.group(1) or m.group(2) or m.group(3).startswith('"'):
                continue
            toks = tokenize(m.group(3))
            if len(toks) != 1 or len(toks[0]) < 4 or vocab.get(toks[0], 0) > 0:
                continue
            tok, limit = toks[0], (1 if len(toks[0]) <= 5 else 2)
            best = min(((_edit_distance(tok, t, limit), -c, t) for t, c in vocab.items()), default=None)
            if best and best[0] <= limit:
                pieces += [query[last:m.start(3)], best[2]]
                last, changed = m.end(3), True
        return ("".join(pieces) + query[last:]) if changed else None

    def explain(self, query: str, doc_id: str) -> dict:
        """Break the BM25 score of one document into its per-term contributions."""
        pq = parse_query(query)
        doc = self.get_document(doc_id, with_text=False)
        n, total_len = self._corpus()
        avgdl = total_len / n if n else 1.0
        tfs = {p["term"]: p["tf"] for p in self.postings.find({"doc_id": doc_id, "term": {"$in": pq.terms}}, {"_id": 0})} if pq.terms else {}
        df = self._df(pq.terms)
        title_terms, kw_terms = set(tokenize(doc["title"])), {t for k in doc["keywords"] for t in tokenize(k)}
        dl = doc["length"] or 1
        norm = K1 * (1 - B + B * dl / avgdl)
        rows = []
        for t in pq.terms:
            tf, d = tfs.get(t, 0), df.get(t, 0)
            idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
            part = tf * (K1 + 1) / (tf + norm) if tf else 0.0
            rows.append({"term": t, "tf": tf, "df": d, "idf": round(idf, 4), "tf_part": round(part, 4),
                         "score": round(idf * part, 4), "in_title": t in title_terms, "in_keywords": t in kw_terms})
        return {"doc_id": doc_id, "title": doc["title"], "query": pq.describe(), "rows": rows,
                "score": round(sum(r["score"] for r in rows), 4), "doc_length": dl, "avg_length": round(avgdl, 1),
                "corpus_size": n, "k1": K1, "b": B, "title_boost": TITLE_BOOST, "keyword_boost": KEYWORD_BOOST}

    # ---- reading ----------------------------------------------------------
    def read(self, doc_id: str, page: int = 1, page_size: int = 1000) -> dict:
        if page_size < 50:
            raise DocError("page size must be at least 50 characters")
        doc = self.get_document(doc_id)
        pages = split_pages(doc["text"], page_size)
        if not 1 <= page <= len(pages):
            raise DocError(f"page {page} out of range (document has {len(pages)} page(s))")
        return {"doc_id": doc_id, "title": doc["title"], "type": doc["type"], "year": doc["year"],
                "authors": doc["authors"], "has_file": doc["file_id"] is not None, "filename": doc["filename"],
                "page": page, "pages": len(pages), "content": pages[page - 1]}

    # ---- vectors / comparison / association ---------------------------------
    def _vectors(self, doc_ids) -> dict[str, dict[str, int]]:
        vecs: dict[str, dict[str, int]] = {d: {} for d in doc_ids}
        for p in self.postings.find({"doc_id": {"$in": list(doc_ids)}}, {"_id": 0}):
            vecs[p["doc_id"]][p["term"]] = p["tf"]
        return vecs

    def _tfidf(self, tfs: dict[str, int], n: int, df: dict[str, int]) -> dict[str, float]:
        return {t: (1 + math.log(tf)) * self._idf_smooth(n, df.get(t, 1)) for t, tf in tfs.items()}

    def compare(self, id_a: str, id_b: str, top_terms: int = 10) -> dict:
        a, b = self.get_document(id_a, with_text=False), self.get_document(id_b, with_text=False)
        vecs = self._vectors([id_a, id_b])
        n, _ = self._corpus()
        df = self._df(set(vecs[id_a]) | set(vecs[id_b]))
        wa, wb = self._tfidf(vecs[id_a], n, df), self._tfidf(vecs[id_b], n, df)
        shared = sorted(set(wa) & set(wb), key=lambda t: -wa[t] * wb[t])
        only_a = sorted(set(wa) - set(wb), key=lambda t: -wa[t])
        only_b = sorted(set(wb) - set(wa), key=lambda t: -wb[t])

        def fold(xs):
            return {x.casefold() for x in xs}
        return {
            "a": {"doc_id": id_a, "title": a["title"], "type": a["type"], "year": a["year"]},
            "b": {"doc_id": id_b, "title": b["title"], "type": b["type"], "year": b["year"]},
            "cosine": round(_cosine(wa, wb), 4),
            "shared_terms": [(t, round(wa[t] * wb[t], 3)) for t in shared[:top_terms]],
            "shared_term_count": len(shared),
            "only_a": [(t, round(wa[t], 3)) for t in only_a[:top_terms]],
            "only_b": [(t, round(wb[t], 3)) for t in only_b[:top_terms]],
            "metadata": {
                "same_type": a["type"] == b["type"],
                "year_gap": abs(a["year"] - b["year"]) if a["year"] and b["year"] else None,
                "shared_authors": [x for x in a["authors"] if x.casefold() in fold(b["authors"])],
                "author_jaccard": round(_jaccard(a["authors"], b["authors"]), 3),
                "shared_keywords": [x for x in a["keywords"] if x.casefold() in fold(b["keywords"])],
                "keyword_jaccard": round(_jaccard(a["keywords"], b["keywords"]), 3),
            },
        }

    def related(self, doc_id: str, limit: int = 5, probe_terms: int = 25, max_candidates: int = 200) -> list[dict]:
        """Documents most similar to `doc_id` by TF-IDF cosine. Candidates come from the
        postings of the document's heaviest terms; they are then scored exactly."""
        src = self.get_document(doc_id, with_text=False)
        n, _ = self._corpus()
        tfs = self._vectors([doc_id])[doc_id]
        df = self._df(tfs)
        w_src = self._tfidf(tfs, n, df)
        probe = sorted(w_src, key=lambda t: -w_src[t])[:probe_terms]
        partial: Counter = Counter()
        for p in self.postings.find({"term": {"$in": probe}, "doc_id": {"$ne": doc_id}}, {"_id": 0}):
            partial[p["doc_id"]] += w_src[p["term"]] * (1 + math.log(p["tf"]))
        cand = [d for d, _ in partial.most_common(max_candidates)]
        if not cand:
            return []
        vecs = self._vectors(cand)
        cand_terms = {t for v in vecs.values() for t in v}
        df_all = {**df, **self._df(cand_terms - set(df))}
        scored = sorted(((_cosine(w_src, self._tfidf(vecs[c], n, df_all)), c) for c in cand), reverse=True)[:limit]
        info = {d["_id"]: d for d in self.documents.find({"_id": {"$in": [c for _, c in scored]}}, {"text": 0})}
        out = []
        for sim, c in scored:
            if sim <= 0:
                continue
            d = info[c]
            shared_kw = [k for k in src["keywords"] if k in d["keywords"]]
            shared_t = sorted(set(vecs[c]) & set(w_src), key=lambda t: -w_src[t])[:5]
            out.append({"doc_id": c, "title": d["title"], "type": d["type"], "year": d["year"],
                        "similarity": round(sim, 4), "shared_terms": shared_t, "shared_keywords": shared_kw})
        return out

    def similarity_graph(self, threshold: float = 0.25, doc_type: str | None = None) -> dict:
        """All-pairs TF-IDF cosine over the postings, keeping edges >= threshold.
        Cost is O(sum over terms of df^2): fine for thousands of documents; use
        `related` or restrict with doc_type for much larger corpora."""
        flt = {"type": _norm_type(doc_type)} if doc_type else {}
        nodes = {d["_id"]: d for d in self.documents.find(flt, {"title": 1, "type": 1, "year": 1})}
        n, _ = self._corpus()
        norm2: dict[str, float] = defaultdict(float)
        dots: dict[tuple[str, str], float] = defaultdict(float)
        cursor = self.postings.find({}, {"_id": 0}).sort([("term", ASCENDING)])
        for _term, group in groupby(cursor, key=lambda p: p["term"]):
            plist = list(group)
            idf = self._idf_smooth(n, len(plist))
            ws = [(p["doc_id"], (1 + math.log(p["tf"])) * idf) for p in plist if p["doc_id"] in nodes]
            for _d, w in ws:
                norm2[_d] += w * w
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    (da, wa), (db, wb) = ws[i], ws[j]
                    dots[(da, db) if da < db else (db, da)] += wa * wb
        edges = []
        for (da, db), dot in dots.items():
            sim = dot / math.sqrt(norm2[da] * norm2[db])
            if sim >= threshold:
                edges.append({"source": da, "target": db, "weight": round(sim, 4)})
        edges.sort(key=lambda e: (-e["weight"], e["source"], e["target"]))

        parent = {d: d for d in nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for e in edges:
            parent[find(e["source"])] = find(e["target"])
        comps: dict[str, list[str]] = defaultdict(list)
        for d in nodes:
            comps[find(d)].append(d)
        components = sorted((sorted(c) for c in comps.values()), key=lambda c: (-len(c), c))
        return {"threshold": threshold,
                "nodes": [{"id": d, "title": v["title"], "type": v["type"], "year": v["year"]} for d, v in sorted(nodes.items())],
                "edges": edges, "components": components}

    def _pair_counts(self, path: str, min_count: int = 1, only: str | None = None, doc_type: str | None = None,
                     limit: int = 50) -> list[dict]:
        """Pairs of values of an array field (keywords, authors) found in the same document,
        counted by one aggregation pipeline inside MongoDB."""
        match = {"type": _norm_type(doc_type)} if doc_type else {}
        pipeline = [
            {"$match": match},
            {"$project": {"a": f"${path}", "b": f"${path}"}},
            {"$unwind": "$a"}, {"$unwind": "$b"},
            {"$match": {"$expr": {"$lt": ["$a", "$b"]}}},
        ]
        if only:
            pipeline.append({"$match": {"$or": [{"a": only}, {"b": only}]}})
        pipeline += [
            {"$group": {"_id": {"a": "$a", "b": "$b"}, "count": {"$sum": 1}, "docs": {"$push": "$_id"}}},
            {"$match": {"count": {"$gte": int(min_count)}}},
            {"$sort": {"count": -1, "_id.a": 1, "_id.b": 1}},
            {"$limit": int(limit)},
        ]
        return [{"a": r["_id"]["a"], "b": r["_id"]["b"], "count": r["count"], "docs": sorted(r["docs"])}
                for r in self.documents.aggregate(pipeline)]

    def keyword_cooccurrence(self, min_count: int = 1, keyword: str | None = None,
                             doc_type: str | None = None, limit: int = 50) -> list[dict]:
        """Keyword pairs that appear together in the same document."""
        return self._pair_counts("keywords", min_count, keyword.strip().lower() if keyword else None, doc_type, limit)

    def author_network(self, min_count: int = 1, author: str | None = None, limit: int = 60) -> list[dict]:
        """Co-authorship: author pairs with the number of documents they share."""
        return self._pair_counts("authors", min_count, author.strip() if author else None, None, limit)

    def analytics(self) -> dict:
        """Collection overview: documents per year and type, most productive authors, keywords, co-authorship."""
        by_year: dict[int, dict] = {}
        for r in self.documents.aggregate([{"$match": {"year": {"$ne": None}}},
                                           {"$group": {"_id": {"y": "$year", "t": "$type"}, "n": {"$sum": 1}}}]):
            row = by_year.setdefault(r["_id"]["y"], {"year": r["_id"]["y"], "total": 0, **{t: 0 for t in DOC_TYPES}})
            row[r["_id"]["t"]] = r["n"]
            row["total"] += r["n"]

        def top(path, limit):
            return [(r["_id"], r["n"]) for r in self.documents.aggregate(
                [{"$unwind": f"${path}"}, {"$group": {"_id": f"${path}", "n": {"$sum": 1}}},
                 {"$sort": {"n": -1, "_id": 1}}, {"$limit": limit}])]
        return {"by_year": [by_year[y] for y in sorted(by_year)], "top_authors": top("authors", 30),
                "top_keywords": top("keywords", 15), "author_edges": self.author_network(limit=60),
                "documents": self._corpus()[0]}

    def insights(self, doc_id: str, n_terms: int = 10, n_sentences: int = 3) -> dict:
        """Automatic reading aids: key terms (TF-IDF), a short extractive summary, reading time."""
        doc = self.get_document(doc_id)
        n, _ = self._corpus()
        tfs = self._vectors([doc_id])[doc_id]
        w = self._tfidf(tfs, n, self._df(tfs))
        ranked = sorted(w, key=lambda t: (-w[t], t))
        key_terms = [(t, round(w[t], 3)) for t in ranked[:n_terms]]
        text = re.sub(r"\s+", " ", doc["text"]).strip()
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) >= 6]
        if len(sentences) <= n_sentences:
            summary = sentences
        else:
            def score(i):
                toks = set(tokenize(sentences[i]))
                return sum(w.get(t, 0) for t in toks) / math.sqrt(len(toks) or 1) * (1.2 if i == 0 else 1.0)
            best = sorted(range(len(sentences)), key=lambda i: -score(i))[:n_sentences]
            summary = [sentences[i] for i in sorted(best)]
        have = {t for k in doc["keywords"] for t in tokenize(k)}
        words = len(text.split())
        return {"doc_id": doc_id, "key_terms": key_terms, "summary": summary, "words": words,
                "reading_minutes": max(1, round(words / 200)), "distinct_terms": len(tfs),
                "suggested_keywords": [t for t, _ in key_terms if t not in have][:5]}

    def find_duplicates(self, threshold: float = 0.6, doc_type: str | None = None) -> list[dict]:
        """Pairs of documents whose TF-IDF cosine similarity is at least `threshold`."""
        g = self.similarity_graph(threshold, doc_type)
        info = {x["id"]: x for x in g["nodes"]}
        return [{"a": info[e["source"]], "b": info[e["target"]], "similarity": e["weight"]} for e in g["edges"]]

    def citation(self, doc_id: str, fmt: str = "bibtex") -> str:
        """The document as a BibTeX, RIS or APA-style reference."""
        d = self.get_document(doc_id, with_text=False)
        fmt = fmt.lower()
        if fmt == "bibtex":
            kind = {"paper": "article", "patent": "misc", "report": "techreport", "drawing": "misc"}[d["type"]]
            key = re.sub(r"[^A-Za-z0-9]+", "_", doc_id)
            lines = [f"@{kind}{{{key},"]
            if d["authors"]:
                lines.append("  author = {" + " and ".join(_bibtex_escape(a) for a in d["authors"]) + "},")
            lines.append("  title = {{" + _bibtex_escape(d["title"]) + "}},")
            if d["year"]:
                lines.append(f"  year = {{{d['year']}}},")
            if d["keywords"]:
                lines.append("  keywords = {" + ", ".join(_bibtex_escape(k) for k in d["keywords"]) + "},")
            if d["type"] in ("patent", "drawing"):
                lines.append("  howpublished = {" + ("Patent" if d["type"] == "patent" else "Engineering drawing") + "},")
            lines.append("}")
            return "\n".join(lines) + "\n"
        if fmt == "ris":
            ty = {"paper": "JOUR", "patent": "PAT", "report": "RPRT", "drawing": "GEN"}[d["type"]]
            lines = [f"TY  - {ty}", f"ID  - {doc_id}", f"TI  - {d['title']}"]
            lines += [f"AU  - {a}" for a in d["authors"]]
            if d["year"]:
                lines.append(f"PY  - {d['year']}")
            lines += [f"KW  - {k}" for k in d["keywords"]]
            if d["abstract"]:
                lines.append(f"AB  - {d['abstract']}")
            lines.append("ER  - ")
            return "\r\n".join(lines) + "\r\n"
        if fmt == "apa":
            def initials(name):
                parts = name.replace(",", " ").split()
                return f"{parts[-1]}, " + " ".join(p[0].upper() + "." for p in parts[:-1]) if len(parts) > 1 else name
            names = [initials(a) for a in d["authors"]]
            who = names[0] if len(names) == 1 else (", ".join(names[:-1]) + ", & " + names[-1] if names else "Anonymous")
            label = {"paper": "", "patent": " [Patent]", "report": " [Technical report]", "drawing": " [Engineering drawing]"}[d["type"]]
            return f"{who} ({d['year'] or 'n.d.'}). {d['title']}{label}. {doc_id}.\n"
        raise DocError(f"unknown citation format {fmt!r}; use bibtex, ris or apa")

    # ---- download ---------------------------------------------------------
    def download(self, doc_id: str, dest_dir: str = "downloads") -> dict:
        """Write the original file from GridFS into dest_dir (never overwrites an
        existing file: a numeric suffix is added) and verify its checksum."""
        doc = self.get_document(doc_id, with_text=False)
        if doc.get("file_id") is None:
            raise DocError(f"document {doc_id} has no stored file")
        os.makedirs(dest_dir, exist_ok=True)
        base, ext = os.path.splitext(_safe_filename(doc["filename"]))
        path, i = os.path.join(dest_dir, base + ext), 1
        while os.path.exists(path):
            path = os.path.join(dest_dir, f"{base} ({i}){ext}")
            i += 1
        h = hashlib.sha256()
        try:
            with open(path, "wb") as out:
                stream = self.fs.open_download_stream(doc["file_id"])
                for chunk in iter(lambda: stream.read(1 << 16), b""):
                    h.update(chunk)
                    out.write(chunk)
        except gridfs.errors.NoFile:
            if os.path.exists(path):
                os.remove(path)
            raise DocError(f"GridFS file for {doc_id} is missing")
        if doc.get("sha256") and h.hexdigest() != doc["sha256"]:
            os.remove(path)
            raise DocError(f"checksum mismatch while downloading {doc_id}")
        return {"doc_id": doc_id, "path": os.path.abspath(path), "size": os.path.getsize(path),
                "sha256": h.hexdigest(), "content_type": doc["content_type"]}

    # ---- stats ------------------------------------------------------------
    def stats(self) -> dict:
        n, total_len = self._corpus()
        by_type = {r["_id"]: r["n"] for r in self.documents.aggregate(
            [{"$group": {"_id": "$type", "n": {"$sum": 1}}}, {"$sort": {"_id": 1}}])}
        files = list(self.db["fs.files"].aggregate([{"$group": {"_id": None, "n": {"$sum": 1}, "bytes": {"$sum": "$length"}}}]))
        distinct = list(self.postings.aggregate([{"$group": {"_id": "$term"}}, {"$count": "n"}], allowDiskUse=True))
        top_terms = [(r["_id"], r["df"]) for r in self.postings.aggregate(
            [{"$group": {"_id": "$term", "df": {"$sum": 1}}}, {"$sort": {"df": -1, "_id": 1}}, {"$limit": 8}], allowDiskUse=True)]
        top_kw = [(r["_id"], r["n"]) for r in self.documents.aggregate(
            [{"$unwind": "$keywords"}, {"$group": {"_id": "$keywords", "n": {"$sum": 1}}},
             {"$sort": {"n": -1, "_id": 1}}, {"$limit": 8}])]
        years = list(self.documents.aggregate([{"$match": {"year": {"$ne": None}}},
                                               {"$group": {"_id": None, "lo": {"$min": "$year"}, "hi": {"$max": "$year"}}}]))
        authors = list(self.documents.aggregate([{"$unwind": "$authors"}, {"$group": {"_id": "$authors"}}, {"$count": "n"}]))
        return {
            "database": self.db.name, "documents": n, "by_type": by_type,
            "gridfs_files": files[0]["n"] if files else 0, "gridfs_bytes": files[0]["bytes"] if files else 0,
            "postings": self.postings.count_documents({}),
            "distinct_terms": distinct[0]["n"] if distinct else 0,
            "avg_doc_length": round(total_len / n, 1) if n else 0,
            "year_range": (years[0]["lo"], years[0]["hi"]) if years else None,
            "distinct_authors": authors[0]["n"] if authors else 0,
            "top_terms_by_df": top_terms, "top_keywords": top_kw,
        }


# --------------------------------------------------------------------------
# Rendering (CLI output)
# --------------------------------------------------------------------------
def _hdr(title: str) -> None:
    print(f"\n=== {title} ===")


def render_list(docs: list[dict]) -> None:
    if not docs:
        print("(no documents)")
        return
    print(f"{'id':<10} {'type':<8} {'year':<5} {'file':<5} title")
    for d in docs:
        print(f"{d['_id']:<10} {d['type']:<8} {d['year'] or '-':<5} {'yes' if d.get('file_id') else 'no':<5} {d['title']}")
    print(f"({len(docs)} document(s))")


def render_search(results: list[dict]) -> None:
    if not results:
        print("(no matching documents)")
        return
    for i, r in enumerate(results, 1):
        print(f"{i:>2}. [{r['score']:.3f}] {r['doc_id']}  {r['type']}, {r['year'] or 'n/a'}  {r['title']}")
        print(f"     authors: {', '.join(r['authors']) or '-'}")
        if r.get("snippet"):
            print(f"     {r['snippet']}")


def render_search_page(r: dict, offset: int = 0) -> None:
    q = r["query"]
    parts = [f"terms: {' '.join(q['terms']) or '-'}"]
    if q["phrases"]:
        parts.append("phrases: " + ", ".join(f'"{p}"' for p in q["phrases"]))
    if q["exclude"]:
        parts.append("excluding: " + " ".join(q["exclude"]))
    parts += q["filters"]
    print(f"{r['total']} match(es)   [{'; '.join(parts)}]")
    if r["suggestion"]:
        print(f"Did you mean: {r['suggestion']}")
    render_search(r["results"]) if r["results"] else print("(no matching documents)")
    shown = offset + len(r["results"])
    if r["total"] > shown:
        print(f"... {r['total'] - shown} more (use --offset {shown})")
    if r["total"]:
        f = r["facets"]
        print("facets  type: " + ", ".join(f"{k}={n}" for k, n in f["type"])
              + "  | year: " + ", ".join(f"{k}={n}" for k, n in f["year"][:6])
              + "  | author: " + ", ".join(f"{k}={n}" for k, n in f["author"][:4]))


def render_explain(e: dict) -> None:
    print(f"{e['doc_id']}  {e['title']}")
    print(f"BM25 score {e['score']:.4f}   (N={e['corpus_size']}, doc length {e['doc_length']}, avg length {e['avg_length']}, k1={e['k1']}, b={e['b']})")
    print(f"{'term':<14}{'tf':>4}{'df':>4}{'idf':>8}{'tf part':>9}{'score':>9}  where")
    for r in e["rows"]:
        where = ", ".join(w for w, on in (("title", r["in_title"]), ("keywords", r["in_keywords"])) if on) or "text"
        print(f"{r['term']:<14}{r['tf']:>4}{r['df']:>4}{r['idf']:>8.3f}{r['tf_part']:>9.3f}{r['score']:>9.3f}  {where if r['tf'] else 'absent'}")
    print(f"(tf counts a title occurrence x{e['title_boost']} and a keyword occurrence x{e['keyword_boost']}.)")


def render_insights(i: dict) -> None:
    print(f"{i['doc_id']}: {i['words']} words, about {i['reading_minutes']} min read, {i['distinct_terms']} distinct index terms")
    print("key terms: " + ", ".join(f"{t} ({w})" for t, w in i["key_terms"]))
    if i["suggested_keywords"]:
        print("could be keywords: " + ", ".join(i["suggested_keywords"]))
    print("summary:")
    for x in i["summary"]:
        print(f"  - {x}")


def render_read(r: dict) -> None:
    print(f"{r['title']}  [{r['doc_id']}, {r['type']}, {r['year'] or 'n/a'}]")
    print(f"authors: {', '.join(r['authors']) or '-'}")
    if r["has_file"]:
        print(f"original file: {r['filename']} (use 'download {r['doc_id']}' to fetch it)")
    print(f"--- page {r['page']} of {r['pages']} ---")
    print(r["content"].strip())
    print(f"--- end of page {r['page']}/{r['pages']} ---")


def render_compare(c: dict) -> None:
    a, b, m = c["a"], c["b"], c["metadata"]
    print(f"A: {a['doc_id']}  {a['title']} ({a['type']}, {a['year']})")
    print(f"B: {b['doc_id']}  {b['title']} ({b['type']}, {b['year']})")
    print(f"TF-IDF cosine similarity: {c['cosine']:.4f}")
    print(f"Shared terms ({c['shared_term_count']} total, top by joint weight): "
          + (", ".join(t for t, _ in c["shared_terms"]) or "-"))
    print("Only in A: " + (", ".join(t for t, _ in c["only_a"]) or "-"))
    print("Only in B: " + (", ".join(t for t, _ in c["only_b"]) or "-"))
    print(f"Metadata: same type={m['same_type']}, year gap={m['year_gap']}")
    print(f"  shared authors ({m['author_jaccard']:.2f} Jaccard): {', '.join(m['shared_authors']) or '-'}")
    print(f"  shared keywords ({m['keyword_jaccard']:.2f} Jaccard): {', '.join(m['shared_keywords']) or '-'}")


def render_related(src: str, rows: list[dict]) -> None:
    if not rows:
        print(f"(nothing related to {src})")
    for r in rows:
        print(f"{r['similarity']:.4f}  {r['doc_id']}  {r['type']}, {r['year']}  {r['title']}")
        print(f"        shared terms: {', '.join(r['shared_terms']) or '-'}; shared keywords: {', '.join(r['shared_keywords']) or '-'}")


def render_graph(g: dict) -> None:
    print(f"threshold {g['threshold']}: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
    for e in g["edges"]:
        print(f"  {e['source']} -- {e['target']}  {e['weight']:.4f}")
    multi = [c for c in g["components"] if len(c) > 1]
    print(f"connected groups with 2+ documents: {len(multi)}")
    for i, c in enumerate(multi, 1):
        print(f"  group {i}: {', '.join(c)}")
    singles = [c[0] for c in g["components"] if len(c) == 1]
    if singles:
        print(f"  isolated: {', '.join(singles)}")


def graph_to_dot(g: dict) -> str:
    lines = ["graph similarity {", "  node [shape=box, fontsize=10];"]
    for n in g["nodes"]:
        label = n["title"].replace('"', "'")
        lines.append(f'  "{n["id"]}" [label="{n["id"]}\\n{label[:40]}"];')
    for e in g["edges"]:
        lines.append(f'  "{e["source"]}" -- "{e["target"]}" [label="{e["weight"]:.2f}"];')
    lines.append("}")
    return "\n".join(lines)


def render_cooccur(rows: list[dict]) -> None:
    if not rows:
        print("(no keyword pairs)")
    for r in rows:
        print(f"{r['count']:>3}  {r['a']} + {r['b']}   ({', '.join(r['docs'])})")


def render_stats(s: dict) -> None:
    print(f"database: {s['database']}")
    print(f"documents: {s['documents']}  by type: " + (", ".join(f"{k}={v}" for k, v in s["by_type"].items()) or "-"))
    print(f"GridFS files: {s['gridfs_files']} ({s['gridfs_bytes']:,} bytes)")
    print(f"index: {s['postings']:,} postings, {s['distinct_terms']:,} distinct terms, avg doc length {s['avg_doc_length']}")
    print(f"years: {s['year_range'] or '-'}, distinct authors: {s['distinct_authors']}")
    print("most widespread terms (df): " + ", ".join(f"{t}={n}" for t, n in s["top_terms_by_df"]))
    print("most common keywords: " + ", ".join(f"{t}={n}" for t, n in s["top_keywords"]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _split_multi(values, seps) -> list[str]:
    out = []
    for v in values or []:
        out += re.split(seps, v)
    return [x.strip() for x in out if x.strip()]


def cmd_add(st, a):
    text = a.text or ""
    if a.text_file:
        with open(a.text_file, encoding="utf-8") as fh:
            text = fh.read()
    r = st.add_document(a.title, a.type, authors=_split_multi(a.author, r";"), keywords=_split_multi(a.keyword, r","),
                        year=a.year, abstract=a.abstract or "", text=text, file_path=a.file, doc_id=a.doc_id)
    print(f"stored {r['doc_id']} ({r['type']}): {r['terms']} index terms" + (f", file {r['size']:,} bytes in GridFS" if r["size"] is not None else ", no file"))


def cmd_list(st, a):
    render_list(st.list_documents(a.type, a.limit))


def cmd_search(st, a):
    r = st.search_ex(" ".join(a.query), a.type, a.year_from, a.year_to, a.author, a.limit, a.offset,
                     require_all=a.all, sort=a.sort)
    render_search_page(r, a.offset)


def cmd_explain(st, a):
    render_explain(st.explain(" ".join(a.query), a.doc_id))


def cmd_insights(st, a):
    render_insights(st.insights(a.doc_id, n_sentences=a.sentences))


def cmd_cite(st, a):
    print(st.citation(a.doc_id, a.format), end="")


def cmd_duplicates(st, a):
    pairs = st.find_duplicates(a.threshold)
    if not pairs:
        print(f"(no pairs with cosine >= {a.threshold})")
    for p in pairs:
        print(f"{p['similarity']:.4f}  {p['a']['id']} {p['a']['title'][:40]}  <->  {p['b']['id']} {p['b']['title'][:40]}")


def cmd_authors(st, a):
    an = st.analytics()
    print("documents per year: " + ", ".join(f"{r['year']}={r['total']}" for r in an["by_year"]))
    print("top authors: " + ", ".join(f"{n} ({c})" for n, c in an["top_authors"][:8]))
    print("co-authorship (pairs sharing documents):")
    for e in st.author_network(a.min_count, a.author, a.limit):
        print(f"  {e['count']:>2}  {e['a']} + {e['b']}   ({', '.join(e['docs'])})")


def cmd_read(st, a):
    render_read(st.read(a.doc_id, a.page, a.page_size))


def cmd_compare(st, a):
    render_compare(st.compare(a.doc_a, a.doc_b, a.top))


def cmd_related(st, a):
    render_related(a.doc_id, st.related(a.doc_id, a.limit))


def cmd_graph(st, a):
    g = st.similarity_graph(a.threshold, a.type)
    render_graph(g)
    if a.dot:
        with open(a.dot, "w", encoding="utf-8") as fh:
            fh.write(graph_to_dot(g))
        print(f"wrote Graphviz file {a.dot}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(g, fh, indent=2)
        print(f"wrote JSON {a.json}")


def cmd_cooccur(st, a):
    render_cooccur(st.keyword_cooccurrence(a.min_count, a.keyword, a.type, a.limit))


def cmd_download(st, a):
    r = st.download(a.doc_id, a.dest)
    print(f"wrote {r['path']} ({r['size']:,} bytes, sha256 {r['sha256'][:16]}..., checksum verified)")


def cmd_delete(st, a):
    for d in a.doc_id:
        r = st.delete_document(d)
        print(f"deleted {d}: {r['postings_removed']} postings removed, GridFS file {'removed' if r['file_removed'] else 'none'}")


def cmd_stats(st, a):
    render_stats(st.stats())


def cmd_demo(st, a):
    import sample_data
    sample = sample_data.documents()
    _hdr(f"1. Storage: loading {len(sample)} sample documents into database '{st.db.name}'")
    for s in sample:
        try:
            st.delete_document(s["doc_id"])  # keep the demo repeatable
        except NotFound:
            pass
        r = st.add_document(**s)
        print(f"  stored {r['doc_id']:<8} {r['type']:<8} {s['title'][:58]:<58} "
              + (f"file {r['size']:>6,} B" if r["size"] else ""))
    render_stats(st.stats())

    _hdr("2. List")
    render_list(st.list_documents())

    _hdr("3. Ranked search (BM25): 'inverted index compression'")
    render_search(st.search("inverted index compression", limit=4))
    _hdr("3b. Filtered search: 'vibration fault detection', type=patent")
    render_search(st.search("vibration fault detection", doc_type="patent"))
    _hdr("3c. Filtered search: 'nosql cluster', years 2021-2023, author 'haddad'")
    render_search(st.search("nosql cluster", year_from=2021, year_to=2023, author="haddad"))

    _hdr("4. Reading (paginated, 450 characters per page): pap-003, page 2")
    render_read(st.read("pap-003", page=2, page_size=450))

    _hdr("5. Comparison")
    render_compare(st.compare("pap-003", "rep-002"))
    print()
    render_compare(st.compare("pap-003", "pat-001"))

    _hdr("6. Association analysis: related documents for pat-002")
    render_related("pat-002", st.related("pat-002", limit=4))
    _hdr("6b. Similarity graph (cosine >= 0.10)")
    render_graph(st.similarity_graph(0.10))
    _hdr("6c. Keyword co-occurrence (pairs seen in 2+ documents)")
    render_cooccur(st.keyword_cooccurrence(min_count=2, limit=12))

    _hdr("7. Download an engineering drawing from GridFS")
    cmd_download(st, argparse.Namespace(doc_id="drw-001", dest=a.dest))

    _hdr("8. Delete cleanup: add a throwaway document with a file, then delete it")
    tmp = st.add_document("Temporary scratch note on inverted index experiments", "report", authors=["Demo User"],
                          keywords=["scratch"], year=2025, text="inverted index scratch note " * 5,
                          file_bytes=b"scratch bytes", filename="scratch.txt", doc_id="tmp-demo")
    before = st.stats()
    print(f"after adding {tmp['doc_id']}: {before['documents']} docs, {before['postings']} postings, {before['gridfs_files']} GridFS files")
    cmd_delete(st, argparse.Namespace(doc_id=["tmp-demo"]))
    after = st.stats()
    print(f"after deleting it : {after['documents']} docs, {after['postings']} postings, {after['gridfs_files']} GridFS files")
    left = st.postings.count_documents({"doc_id": "tmp-demo"})
    print(f"postings still pointing at tmp-demo: {left}")

    _hdr('9. Query language: "exact phrase", -exclusion, field filters')
    for q in ('"posting list"', "vibration -federated", "type:report year:>=2024", 'keyword:"wind turbine" author:berg fault'):
        print(f"query: {q}")
        render_search_page(st.search_ex(q, limit=3, snippet=False))
        print()
    _hdr("9b. Spelling suggestion")
    render_search_page(st.search_ex("gearbxo vibraton", limit=3, snippet=False))
    _hdr("9c. Why this score? BM25 breakdown of pat-002 for 'vibration sensor'")
    render_explain(st.explain("vibration sensor", "pat-002"))
    _hdr("9d. Document insights: pap-003")
    render_insights(st.insights("pap-003"))
    _hdr("9e. Citation export: pap-003 as BibTeX")
    print(st.citation("pap-003", "bibtex"), end="")
    _hdr("9f. Analytics: authors and co-authorship")
    cmd_authors(st, argparse.Namespace(min_count=1, author=None, limit=6))
    _hdr("9g. Near-duplicate check (cosine >= 0.20)")
    cmd_duplicates(st, argparse.Namespace(threshold=0.20))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docretrieval", description="Large-scale document retrieval system on MongoDB + GridFS")
    p.add_argument("--uri", default=os.environ.get("MONGO_URI", DEFAULT_URI), help=f"MongoDB URI (default {DEFAULT_URI}, or $MONGO_URI)")
    p.add_argument("--db", default=None, help=f"database name (default {DEFAULT_DB}, or $DOCRETRIEVAL_DB; the demo uses {DEMO_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="store a document (metadata, text, optional original file)")
    s.add_argument("--title", required=True)
    s.add_argument("--type", required=True, help="paper | patent | report | drawing")
    s.add_argument("--author", action="append", help="repeatable; ';' also separates names")
    s.add_argument("--keyword", action="append", help="repeatable; ',' also separates keywords")
    s.add_argument("--year", type=int)
    s.add_argument("--abstract")
    s.add_argument("--text", help="full text given on the command line")
    s.add_argument("--text-file", help="read full text from this UTF-8 file")
    s.add_argument("--file", help="original file to keep in GridFS (PDF, TXT, PNG, DXF, ...)")
    s.add_argument("--doc-id", help="explicit id (default: generated)")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("list", help="list stored documents")
    s.add_argument("--type")
    s.add_argument("--limit", type=int, default=100)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("search", help="BM25 ranked search with filters and snippets")
    s.add_argument("query", nargs="+")
    s.add_argument("--type")
    s.add_argument("--year-from", type=int)
    s.add_argument("--year-to", type=int)
    s.add_argument("--author", help="case-insensitive substring of an author name")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--all", action="store_true", help="require every query term (default: any)")
    s.add_argument("--offset", type=int, default=0, help="skip this many results (paging)")
    s.add_argument("--sort", choices=("relevance", "year"), default="relevance")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("explain", help="show how the BM25 score of a document is made up for a query")
    s.add_argument("doc_id")
    s.add_argument("query", nargs="+")
    s.set_defaults(fn=cmd_explain)

    s = sub.add_parser("insights", help="key terms, extractive summary and reading time of a document")
    s.add_argument("doc_id")
    s.add_argument("--sentences", type=int, default=3)
    s.set_defaults(fn=cmd_insights)

    s = sub.add_parser("cite", help="print a BibTeX, RIS or APA reference for a document")
    s.add_argument("doc_id")
    s.add_argument("--format", choices=("bibtex", "ris", "apa"), default="bibtex")
    s.set_defaults(fn=cmd_cite)

    s = sub.add_parser("duplicates", help="list near-duplicate document pairs")
    s.add_argument("--threshold", type=float, default=0.6)
    s.set_defaults(fn=cmd_duplicates)

    s = sub.add_parser("authors", help="documents per year, top authors and co-authorship")
    s.add_argument("--min-count", type=int, default=1)
    s.add_argument("--author")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_authors)

    s = sub.add_parser("read", help="read a document one page at a time")
    s.add_argument("doc_id")
    s.add_argument("--page", type=int, default=1)
    s.add_argument("--page-size", type=int, default=1000, help="characters per page")
    s.set_defaults(fn=cmd_read)

    s = sub.add_parser("compare", help="compare two documents")
    s.add_argument("doc_a")
    s.add_argument("doc_b")
    s.add_argument("--top", type=int, default=10, help="terms to list per group")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("related", help="documents most similar to a given one")
    s.add_argument("doc_id")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(fn=cmd_related)

    s = sub.add_parser("graph", help="similarity graph above a cosine threshold")
    s.add_argument("--threshold", type=float, default=0.25)
    s.add_argument("--type")
    s.add_argument("--dot", help="also write a Graphviz .dot file")
    s.add_argument("--json", help="also write nodes/edges/components as JSON")
    s.set_defaults(fn=cmd_graph)

    s = sub.add_parser("cooccur", help="keyword co-occurrence counts")
    s.add_argument("--min-count", type=int, default=1)
    s.add_argument("--keyword", help="only pairs that include this keyword")
    s.add_argument("--type")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_cooccur)

    s = sub.add_parser("download", help="write the original file from GridFS to a folder")
    s.add_argument("doc_id")
    s.add_argument("--dest", default="downloads")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("delete", help="delete document(s), their index entries and GridFS files")
    s.add_argument("doc_id", nargs="+")
    s.set_defaults(fn=cmd_delete)

    s = sub.add_parser("stats", help="collection, index and storage statistics")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("demo", help="load the sample documents and tour every feature")
    s.add_argument("--dest", default="downloads", help="folder for the demo download")
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    db_name = args.db or os.environ.get("DOCRETRIEVAL_DB") or (DEMO_DB if args.cmd == "demo" else DEFAULT_DB)
    try:
        store = DocumentStore(args.uri, db_name)
    except ServerSelectionTimeoutError:
        print("error: cannot reach MongoDB. Start one, e.g. 'docker run -d -p 27017:27017 mongo' "
              "(see README), or check the connection string, Atlas network access and any VPN/proxy.", file=sys.stderr)
        return 2
    except PyMongoError as e:  # e.g. DNS (SRV) lookup failure for a mongodb+srv:// address
        print(f"error: could not connect to MongoDB ({type(e).__name__}: {str(e)[:160]}). If this mentions DNS, "
              "a VPN/proxy or the network may be blocking the lookup; try another network.", file=sys.stderr)
        return 2
    try:
        args.fn(store, args)
    except DocError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except PyMongoError as e:
        print(f"database error: {e}", file=sys.stderr)
        return 2
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
