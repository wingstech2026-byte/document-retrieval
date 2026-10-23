"""Draws the benchmark charts used in the report from docs/benchmark_results_*.json  ->  docs/figures/*.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
SIZES = (2000, 5000, 10000)
R = {n: json.load(open(os.path.join(HERE, f"benchmark_results_{n}.json"))) for n in SIZES}

names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": "Calibri" if "Calibri" in names else "DejaVu Sans", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a9aa5", "axes.labelcolor": "#12303f",
                     "xtick.color": "#12303f", "ytick.color": "#12303f", "axes.titleweight": "bold", "axes.titlecolor": "#12303f"})
TEAL, AMBER, BLUE, PURPLE, GREEN, BROWN, GREY = "#0E7C7B", "#D98A1C", "#2F6FBD", "#6A4AA8", "#0B8A6B", "#B4631A", "#6b7c86"

# ---- 1. search latency versus collection size ------------------------------------------------------------
series = [("2 rare terms", "search, 2 rare terms", GREEN), ("2 medium terms", "search, 2 medium terms", TEAL),
          ("phrase, 2 medium terms", "phrase search (2 medium terms)", PURPLE), ("type + year filters", "search with type and year filters", BLUE),
          ("author filter", "search with author filter", BROWN), ("2 terms found in every document", "search, 2 frequent terms", AMBER)]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
for label, key, col in series:
    ys = [R[n]["queries"][key]["median_ms"] for n in SIZES]
    ax.plot(SIZES, ys, marker="o", color=col, lw=2.4, ms=6)
    nudge = {"phrase, 2 medium terms": 1.13, "author filter": 0.89}.get(label, 1.0)
    ax.annotate(f"{label}  ({ys[-1]:.0f} ms)", (SIZES[-1], ys[-1] * nudge), xytext=(8, 0), textcoords="offset points", va="center", color=col, fontsize=11, fontweight="bold")
ax.set_yscale("log")
ax.set_yticks([5, 10, 20, 50, 100, 200])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(SIZES)
ax.set_xticklabels([f"{n:,}" for n in SIZES])
ax.set_xlim(1500, 15500)
ax.set_xlabel("documents in the collection")
ax.set_ylabel("median time per search (ms, log scale)")
ax.set_title("Search stays fast for ordinary queries; terms in every document cost the most", loc="left", fontsize=13)
ax.grid(axis="y", color="#e3e9ec", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_search_latency.png"))
plt.close(fig)

# ---- 2. what the index buys ----------------------------------------------------------------------------------------
e = R[10000]["index_effect"]
fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
groups = [f"term in every document\n({e['matching_postings']:,} postings returned)", f"rarer term\n({e['rare_term_postings']:,} postings returned)"]
idx = [e["with_index_ms"], e["rare_with_index_ms"]]
scan = [e["collection_scan_ms"], e["rare_collection_scan_ms"]]
x = [0, 1]
w = 0.34
b1 = ax.bar([i - w / 2 for i in x], idx, w, color=TEAL, label="with the (term, doc_id) index")
b2 = ax.bar([i + w / 2 for i in x], scan, w, color=GREY, label="forced collection scan")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{b.get_height():.0f} ms", ha="center", fontsize=11, fontweight="bold", color="#12303f")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("median time to fetch the postings of one term (ms)")
ax.set_ylim(0, max(scan) * 1.18)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
ax.set_title(f"The index turns a scan of {e['total_postings']:,} postings into a direct lookup", loc="left", fontsize=13)
ax.grid(axis="y", color="#e3e9ec", lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_index_effect.png"))
plt.close(fig)

# ---- 3. functions that read many postings or documents ---------------------------------------------------------------
series = [("related documents", "related documents", PURPLE), ("statistics page", "statistics", GREY),
          ("keyword co-occurrence", "keyword co-occurrence (whole collection)", BLUE), ("compare two documents", "compare two documents", AMBER)]
fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
for label, key, col in series:
    ys = [R[n]["other"][key]["median_ms"] for n in SIZES]
    ax.plot(SIZES, ys, marker="o", color=col, lw=2.4, ms=6)
    ax.annotate(f"{label}  ({ys[-1]:.0f} ms)", (SIZES[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=col, fontsize=11, fontweight="bold")
ax.set_xticks(SIZES)
ax.set_xticklabels([f"{n:,}" for n in SIZES])
ax.set_xlim(1500, 15000)
ax.set_ylim(0, 760)
ax.set_xlabel("documents in the collection")
ax.set_ylabel("median time (ms)")
ax.set_title("Whole-collection functions grow in proportion to the data", loc="left", fontsize=13)
ax.grid(axis="y", color="#e3e9ec", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_scaling_functions.png"))
plt.close(fig)
print("charts written:", sorted(os.listdir(OUT)))
