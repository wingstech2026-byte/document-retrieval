"""Sample corpus for the document retrieval system.

Ten short, original documents across the four types (paper, patent, technical
report, engineering drawing). The people, sites and numbers are invented.
Two topic clusters overlap on purpose so comparison and association analysis
have something to find: database / retrieval systems, and wind-turbine
condition monitoring.

Text documents are stored as UTF-8 .txt files; drawings are small generated PNG
images, i.e. genuinely binary files.
"""
from __future__ import annotations

import math
import struct
import zlib


# --------------------------------------------------------------------------
# Tiny PNG "engineering drawing" generator (grayscale, no dependencies)
# --------------------------------------------------------------------------
class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.px = bytearray([255]) * (w * h)

    def dot(self, x: int, y: int, v: int = 0) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y * self.w + x] = v

    def line(self, x0, y0, x1, y1, v=0, dash: int = 0) -> None:
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for i in range(n + 1):
            if dash and (i // dash) % 2:
                continue
            self.dot(round(x0 + (x1 - x0) * i / n), round(y0 + (y1 - y0) * i / n), v)

    def rect(self, x0, y0, x1, y1, v=0, dash: int = 0) -> None:
        for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            self.line(a, b, c, d, v, dash)

    def circle(self, cx, cy, r, v=0) -> None:
        for deg in range(0, 360 * 4):
            a = math.radians(deg / 4)
            self.dot(round(cx + r * math.cos(a)), round(cy + r * math.sin(a)), v)

    def png(self) -> bytes:
        raw = b"".join(b"\x00" + bytes(self.px[y * self.w:(y + 1) * self.w]) for y in range(self.h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9))
                + chunk(b"IEND", b""))


def gearbox_housing_png() -> bytes:
    c = Canvas(360, 240)
    c.rect(4, 4, 355, 235)                       # sheet border
    c.rect(60, 70, 300, 190)                     # housing outline
    c.rect(75, 85, 285, 175, dash=4)             # hidden inner wall
    for cx in (120, 240):                        # bearing bores
        c.circle(cx, 130, 38)
        c.circle(cx, 130, 14)
    for x in (60, 300):                          # mounting flanges
        c.rect(x - 14 if x < 100 else x, 150, x if x < 100 else x + 14, 190)
    c.line(30, 130, 330, 130, dash=8)            # centre line
    c.rect(230, 205, 350, 230)                   # title block
    return c.png()


def rack_layout_png() -> bytes:
    c = Canvas(360, 240)
    c.rect(4, 4, 355, 235)
    for i in range(3):                           # three racks, one per replica node
        x = 40 + i * 100
        c.rect(x, 30, x + 70, 200)
        for y in range(40, 190, 24):
            c.rect(x + 8, y, x + 62, y + 16)
    c.line(75, 205, 175, 218)                    # patch cabling
    c.line(175, 218, 275, 205)
    c.line(75, 205, 275, 205, dash=5)
    c.rect(230, 215, 350, 232)
    return c.png()


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def _txt(doc_id: str) -> str:
    return f"{doc_id}.txt"


def documents() -> list[dict]:
    """Keyword arguments for DocumentStore.add_document, one dict per document."""
    docs = [
        dict(
            doc_id="pap-001", doc_type="paper", year=2022,
            title="Adaptive Learned Indexes for Skewed Key Distributions",
            authors=["Mei Lin", "Omar Haddad"],
            keywords=["learned index", "b-tree", "query performance", "adaptive structures"],
            abstract="Learned indexes replace tree traversal with a model that predicts where a key lives. "
                     "We show that they degrade when keys cluster heavily, and propose a variant that "
                     "re-fits its model only on the regions that drift.",
            text="""Classic B-tree indexes spend most of their lookup time walking from the root to a leaf. A learned index shortcuts the walk: a small model estimates the position of a key in a sorted array and a short local search corrects the error. On uniformly spread keys the idea works very well, but real workloads such as timestamps, user identifiers and sensor readings are rarely uniform.

We study what happens when keys arrive in dense bursts separated by large gaps. The prediction error of a single global model grows with the sharpest bend in the key distribution, so a handful of hot regions can ruin the lookup cost for the whole structure. Our analysis links the worst-case error to the local curvature of the cumulative distribution of keys.

Our remedy is an adaptive layout. The key space is cut into segments, each with its own tiny model, and a segment is split only when its measured error crosses a threshold. Retraining touches the changed segment rather than the entire index, which keeps insertion cost low.

On three skewed datasets the adaptive layout answered point queries between 1.4 and 2.1 times faster than a tuned B-tree and used a third of the memory. Range queries were slightly slower than the baseline when ranges were short, a trade-off we discuss at the end.""",
        ),
        dict(
            doc_id="pap-002", doc_type="paper", year=2023,
            title="Sharding Strategies for Document Stores under Hotspot Workloads",
            authors=["Omar Haddad", "Priya Nair"],
            keywords=["sharding", "mongodb", "load balancing", "nosql", "hotspots"],
            abstract="Choosing a shard key decides whether a document database scales or stalls. We compare "
                     "hashed, ranged and compound shard keys when a small set of documents receives most of the traffic.",
            text="""Document databases scale out by splitting a collection into chunks and spreading the chunks over several servers. The shard key decides which chunk a document belongs to, and a poor choice concentrates traffic on a single server while the others sit idle. This paper measures that effect on a sharded MongoDB cluster under workloads with a strong hotspot.

A ranged key keeps neighbouring documents together, which helps range scans but sends all new inserts with an increasing key to the last chunk. A hashed key spreads inserts evenly but turns every range query into a scatter-gather across all servers. A compound key that prefixes a low-cardinality field to a high-cardinality one gives a middle path.

We replay traces in which one percent of the documents receive eighty percent of reads. The balancer moves chunks in the background, yet migration cannot split a single hot document, so very hot items need application-level replication or caching.

Our results favour compound keys when queries usually filter on the prefix field, and hashed keys for insert-heavy logging collections. We also report that chunk migration during peak load raised tail latency by roughly a quarter, so we recommend scheduling balancing windows.""",
        ),
        dict(
            doc_id="pap-003", doc_type="paper", year=2024,
            title="Federated Training of Fault Detectors on Wind Turbine Gearboxes",
            authors=["Priya Nair", "Tomas Berg"],
            keywords=["federated learning", "fault detection", "wind turbine", "vibration", "gearbox"],
            abstract="Wind farms rarely share raw sensor data, yet each farm sees too few failures to train a good "
                     "detector alone. We train a vibration-based fault detector across farms without moving the signals.",
            text="""Gearbox failures are among the costliest events in wind power. Vibration sensors on the gearbox housing record the early signs, such as a rising energy band around the meshing frequency of a damaged gear, but each farm observes only a few faults per year. Pooling recordings would help, yet operators treat their sensor data as commercially sensitive.

We therefore train a fault detector in a federated way. Each farm fits a compact neural network on its own vibration windows and sends only model updates to a coordinator, which averages them and returns the improved model. Raw signals never leave the site.

Turbines differ in size, gearbox type and sensor placement, so updates from different farms can disagree. We add a personalisation layer that is kept local, and we weight each farm's contribution by how many labelled fault windows it holds.

On data from five farms the shared detector found bearing and gear faults earlier than detectors trained per farm, raising recall from 0.71 to 0.86 at the same false alarm rate. The approach needs consistent sensor mounting; Section six shows how loose mounts blur the spectrum and hurt the model.""",
        ),
        dict(
            doc_id="pap-004", doc_type="paper", year=2021,
            title="Ranking Short Technical Documents with Sparse and Dense Signals",
            authors=["Mei Lin", "Sofia Alvarez"],
            keywords=["bm25", "ranking", "information retrieval", "inverted index", "technical documents"],
            abstract="Engineering archives are full of short documents with specialised vocabulary. We study how "
                     "classic term-based ranking behaves on them and when adding embedding similarity helps.",
            text="""Archives of standards, memos and drawings notes contain many documents that are only a few hundred words long. Term-based ranking with BM25 handles such collections well because rare technical words are strong evidence, and the inverted index that supports it is cheap to build and update.

Short texts do expose two weaknesses. Length normalisation can over-reward the briefest documents, and a query written with one synonym will miss a document written with another. We examine the length parameter and show that a moderate setting avoids favouring tiny records.

To tackle the vocabulary gap we blend the term score with the similarity of dense embeddings. The blend is only worthwhile for queries that share few words with the target; for queries full of exact part names or identifiers the term score alone is better, and mixing in embeddings sometimes pushed the right answer down.

We therefore propose to route each query: when the query has rare, specific terms, use the inverted index only; otherwise add the dense signal. On a collection of maintenance notes this routing improved the mean reciprocal rank over either method alone.""",
        ),
        dict(
            doc_id="pat-001", doc_type="patent", year=2020,
            title="Method for Compressing Posting Lists Using Block-wise Delta Coding",
            authors=["Sofia Alvarez"],
            keywords=["inverted index", "compression", "posting list", "information retrieval", "delta coding"],
            abstract="A method of storing the document identifiers of an inverted index in fixed-size blocks, "
                     "each coded as differences from its predecessor with a per-block bit width.",
            text="""Claim 1. A method of storing a posting list of an inverted index, the posting list comprising document identifiers in ascending order, the method comprising: dividing the identifiers into blocks of a fixed number of entries; replacing every identifier in a block, except the first, by its difference from the preceding identifier; determining the smallest number of bits able to represent the largest difference in the block; and writing the block as a header holding the first identifier and the bit width, followed by the differences packed at that width.

Claim 2. The method of claim 1, wherein each block header further holds the largest identifier of the block, so that a reader can skip whole blocks while searching for a given identifier without unpacking them.

Claim 3. The method of claim 1, wherein the number of entries per block is chosen so that a block occupies a whole number of cache lines.

Description. Search engines keep, for every term, the list of documents that contain it. These lists dominate the size of the index, and reading them dominates query time. Because identifiers are sorted, the gaps between neighbours are small for common terms, and small numbers need few bits. Choosing the bit width per block lets a few large gaps inflate only their own block instead of the whole list. The invention reduces index size by about forty percent against uncompressed storage while allowing fast skipping during intersection of lists.""",
        ),
        dict(
            doc_id="pat-002", doc_type="patent", year=2022,
            title="Vibration Sensor Mount for Rotating Machinery Housing",
            authors=["Tomas Berg"],
            keywords=["vibration", "sensor", "wind turbine", "gearbox", "mounting"],
            abstract="A clamp that fixes an accelerometer to a curved machine housing with a repeatable preload, "
                     "so readings stay comparable after the sensor is removed and refitted.",
            text="""Claim 1. A mount for an accelerometer on the curved housing of rotating machinery, comprising: a base plate having a concave face matched to the radius of the housing; a stud fixed to the housing through the base plate; a spring washer stack arranged between a nut and the accelerometer body so that tightening the nut to a stop produces a predetermined clamping force; and a locating pin that fixes the angular orientation of the accelerometer relative to the base plate.

Claim 2. The mount of claim 1, wherein the base plate is made of a material whose stiffness is higher than that of the housing, so that the mount does not damp the vibration to be measured.

Description. Condition monitoring of gearboxes relies on comparing the vibration spectrum over weeks and months. Loose or rotated sensors change the measured amplitude and shift resonances, which can look like a developing fault or hide one. Technicians in the field often attach sensors with adhesive or hand-tightened studs, so the preload varies from visit to visit. The spring stack and the locating pin remove both sources of variation. A trial on the gearbox of a wind turbine showed that repeated refitting changed the high-frequency amplitude by less than five percent, compared with more than thirty percent for adhesive mounting.""",
        ),
        dict(
            doc_id="rep-001", doc_type="report", year=2023,
            title="Load Test Report: Replica Set Failover in a Three-Node Document Cluster",
            authors=["Omar Haddad", "Jonas Weber"],
            keywords=["mongodb", "replication", "failover", "nosql", "load testing"],
            abstract="Results of killing the primary of a three-node replica set while a steady write load was running, "
                     "with measured election time, lost writes and client-visible errors.",
            text="""1. Purpose. This report documents a controlled failure test of a three-node replica set used by the document archive. The aim was to learn how long writes stay unavailable after the primary disappears and whether acknowledged writes can be lost.

2. Setup. The nodes ran on separate hosts in one rack. A load generator issued two thousand inserts per second with majority write concern, plus reads from the primary. After ten minutes the primary process was killed without warning. The test was repeated twenty times.

3. Results. A new primary was elected within a median of 4.8 seconds and a worst case of 9.1 seconds. During the gap clients received retryable errors, and with retries enabled no acknowledged write was lost in any run. Without majority write concern, three runs lost between one and six writes that had only reached the old primary.

4. Observations. Election time grew when the network heartbeat timeout was raised from its default, so we recommend keeping the default on a local network. Secondary reads stayed available throughout, with staleness below one second.

5. Recommendation. Use majority write concern for the archive, enable retryable writes in the drivers, and test failover again after every major upgrade.""",
        ),
        dict(
            doc_id="rep-002", doc_type="report", year=2024,
            title="Condition Monitoring Field Report: Gearbox Bearings at Site K",
            authors=["Tomas Berg", "Jonas Weber"],
            keywords=["vibration", "bearings", "fault detection", "wind turbine", "condition monitoring"],
            abstract="Findings from a six-month monitoring campaign on eight turbine gearboxes, including one "
                     "bearing defect found early through changes in the vibration spectrum.",
            text="""1. Scope. Between January and June, accelerometers were installed on the gearbox housings of eight turbines at Site K. Vibration spectra were recorded every ten minutes and summarised into daily trend values for the bearing and gear mesh frequency bands.

2. Method. A baseline was taken during the first four weeks of normal operation. A turbine was flagged when its trend rose more than three standard deviations above its own baseline for five consecutive days. All sensors used the clamp mount described in the company patent so that refitting would not disturb the readings.

3. Findings. Seven gearboxes stayed within their baseline. On turbine 6 the energy near the outer-race frequency of the high-speed shaft bearing began to climb in March. The flag was raised after nine days, and an inspection in April found early spalling on the outer race. The bearing was exchanged during planned maintenance, avoiding an unplanned stop estimated to cost several weeks of generation.

4. Lessons. Per-turbine baselines worked better than a single fleet-wide threshold. Two false alarms were traced to a loose cable rather than the machine. We suggest sharing anonymised spectra between sites so detectors can learn from rare faults.""",
        ),
        dict(
            doc_id="drw-001", doc_type="drawing", year=2023,
            title="Gearbox Housing Assembly Drawing, Sheet 3",
            authors=["Tomas Berg"],
            keywords=["gearbox", "housing", "assembly", "wind turbine", "technical drawing"],
            abstract="Sectional view of the two-stage gearbox housing with bearing bores, mounting flanges and the "
                     "hidden inner wall. Centre line dashed; title block lower right.",
            text="Engineering drawing, sheet 3 of 6. Two-stage gearbox housing, section through both shaft axes. "
                 "Shows the two bearing bores with their retaining shoulders, the left and right mounting flanges, "
                 "the inner wall in dashed hidden-line style and the shaft centre line. Material: cast iron. "
                 "Tolerances for the bearing seats are given in the title block. Sensor boss positions for "
                 "vibration measurement are marked on the upper face.",
            file_bytes=gearbox_housing_png(), filename="gearbox_housing_sheet3.png",
        ),
        dict(
            doc_id="drw-002", doc_type="drawing", year=2023,
            title="Cluster Rack Layout and Cabling Diagram",
            authors=["Jonas Weber"],
            keywords=["rack", "cabling", "cluster", "replication", "nosql"],
            abstract="Front view of the three server racks that host the document database replica nodes, "
                     "with unit positions and patch cabling between racks.",
            text="Layout drawing of three server racks, one per database replica node. Each rack shows its "
                 "equipment units from top to bottom and the patch cabling that links the racks over two "
                 "independent paths so a single cable fault cannot isolate a node. Rack labels, unit numbers "
                 "and cable colours are listed in the title block for the data centre team.",
            file_bytes=rack_layout_png(), filename="cluster_rack_layout.png",
        ),
    ]
    for d in docs:
        if "file_bytes" not in d:
            d["file_bytes"] = d["text"].encode("utf-8")
            d["filename"] = _txt(d["doc_id"])
    return docs


if __name__ == "__main__":
    for d in documents():
        print(f"{d['doc_id']}  {d['doc_type']:<8} {d['year']}  {d['title']}  [{len(d['file_bytes'])} bytes]")
