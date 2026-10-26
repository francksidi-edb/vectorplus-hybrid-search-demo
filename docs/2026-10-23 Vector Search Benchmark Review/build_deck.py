"""Vector search benchmark review: build script (living deck, edb-bento-lite).

Run:  python3 build_deck.py
Every number comes from bench_data.py, which reads the five exports in ./data.
"""
import glob, os, sys
sys.dont_write_bytecode = True
SKILL_DIR = os.environ.get("BENTO_SKILL_DIR")
cands = ([os.path.join(SKILL_DIR, "scripts")] if SKILL_DIR else []) + \
        ["/mnt/skills/plugins/edb-bento-lite/scripts"] + \
        sorted(glob.glob("/mnt/skills/plugins/*edb-bento-lite/scripts") +
               glob.glob("/mnt/skills/user/edb-bento-lite/scripts") +
               [os.path.expanduser("~/.claude/skills/edb-bento-lite/scripts")])
for p in cands:
    if os.path.isfile(os.path.join(p, "bento_lite_v2.py")):
        sys.path.insert(0, p)
        break
else:
    sys.exit("edb-bento-lite library not found; set BENTO_SKILL_DIR to the skill folder")
from bento_lite_v2 import *  # noqa
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.dml import MSO_LINE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT_DIR)
DECK = os.path.join(OUT_DIR, os.path.basename(OUT_DIR) + ".pptx")
import bench_data as B  # noqa

L, R, C = PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER
MID = MSO_ANCHOR.MIDDLE
MONO = "Courier New"

# one colour and one line style per engine, used on every chart
ENG = {
    "w_ivfplus": dict(name="ivfplus WHPG", color=BLUECOAL, dash=None, w=3.0),
    "ivfplus":   dict(name="ivfplus PG", color=BLUECOAL, dash=MSO_LINE.DASH, w=2.25),
    "w_ivfflat": dict(name="IVFFlat WHPG", color=SLATE, dash=None, w=2.25),
    "ivfflat":   dict(name="IVFFlat PG", color=SLATE, dash=MSO_LINE.DASH, w=2.25),
    "vchordrq":  dict(name="VectorChord", color=ORANGE, dash=None, w=2.25),
    "hnsw":      dict(name="pgvector HNSW", color=GREEN, dash=None, w=2.25),
    "clickhouse": dict(name="ClickHouse", color=CORAL, dash=None, w=2.25),
}


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def ms(v):
    return f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"


def gib(b):
    return f"{b / B.GIB:.2f} GiB"


# ------------------------------------------------------------------ building blocks ---
def stat(s, x, y, w, big, cap, size=36, cap_h=0.75, color=BLUECOAL):
    txt(s, x, y, w, 0.66, big, size=size, font=DISPLAY_XB, color=color, bold=True, anchor=MID)
    txt(s, x, y + 0.70, w, cap_h, cap, size=13, color=SLATE)


def rtable(s, x, top, bottom, cols, rows, size=13, head_size=10.5, bold_first=True):
    """Ruled table: caps labels over a Blue Coal rule, Mist hairlines, rows divide the band equally.
    cols: [(label, width, align)]; a cell is a string or (string, color, bold)."""
    total = sum(w for _, w, _ in cols)
    xs = [x]
    for _, w, _ in cols[:-1]:
        xs.append(xs[-1] + w)
    for (label, w, al), cx in zip(cols, xs):
        txt(s, cx, top, w - 0.12, 0.28, label, size=head_size, font=DISPLAY, color=SLATE, caps=True,
            bold=True, align=al, anchor=MID)
    hrule(s, x, top + 0.34, total, color=BLUECOAL, lw=2.0)
    band_top = top + 0.42
    rh = (bottom - band_top) / len(rows)
    for i, row in enumerate(rows):
        y = band_top + i * rh
        for j, ((label, w, al), cx) in enumerate(zip(cols, xs)):
            cell, color, bold = row[j], BLUECOAL, (j == 0 and bold_first)
            if isinstance(cell, tuple):
                t = tuple(cell)
                cell = t[0]
                color = t[1] if len(t) > 1 else BLUECOAL
                bold = t[2] if len(t) > 2 else bold
            txt(s, cx, y, w - 0.12, rh, cell, size=size, color=color, bold=bold, align=al, anchor=MID)
        if i < len(rows) - 1:
            hrule(s, x, y + rh, total, color=MIST, lw=0.75)
    return rh


def code(s, x, y, w, h, lines, size=11):
    panel(s, x, y, w, h, fill=FROST)
    txt(s, x + 0.22, y + 0.16, w - 0.44, h - 0.32, lines, size=size, font=MONO, color=BLUECOAL, spacing=1.12)


def style_axis_text(ax, size=10, color=SLATE):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = color
    ax.tick_labels.font.name = BODY


def axis_title(ax, text, size=10):
    ax.has_title = True
    tf = ax.axis_title.text_frame
    tf.text = text
    for r in tf.paragraphs[0].runs:
        r.font.size = Pt(size)
        r.font.bold = False
        r.font.color.rgb = SLATE
        r.font.name = BODY


def plot_layout(chart, x, y, w, h):
    """Fix the inner plot area (fractions of the chart) so end-of-line labels have room on the right."""
    pa = chart._chartSpace.find(qn("c:chart")).find(qn("c:plotArea"))
    old = pa.find(qn("c:layout"))
    if old is not None:
        pa.remove(old)
    lay = etree.SubElement(pa, qn("c:layout"))
    pa.remove(lay)
    pa.insert(0, lay)
    ml = etree.SubElement(lay, qn("c:manualLayout"))
    for tag, val in (("layoutTarget", "inner"), ("xMode", "edge"), ("yMode", "edge"),
                     ("x", x), ("y", y), ("w", w), ("h", h)):
        e = etree.SubElement(ml, qn("c:" + tag))
        e.set("val", str(val))


def recall_latency_chart(s, x, y, w, h, D, order, label_pos, vline=0.90):
    # LibreOffice drops a scatter series that has fewer points than one drawn after it, so draw the longest
    # series first. Among equals the first name in `order` is drawn last, which puts the hero on top.
    order = sorted(order, key=lambda k: (-len(D[k]["results"]), -order.index(k)))
    cd = XyChartData()
    for k in order:
        sr = cd.add_series(ENG[k]["name"])
        for r in D[k]["results"]:
            sr.add_data_point(r["recall"], r["ms"])
    bar = cd.add_series("bar")
    for yy in (1, 400, 1000):
        bar.add_data_point(vline, yy)
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    tag(s, gf, "chart")
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = BODY
    ch.font.size = Pt(10)
    plot_layout(ch, 0.075, 0.03, 0.74, 0.83)
    xa, ya = ch.category_axis, ch.value_axis
    xa.minimum_scale, xa.maximum_scale, xa.major_unit = 0.2, 1.0, 0.1
    ya.minimum_scale, ya.maximum_scale = 1, 1000
    sc = ya._element.find(qn("c:scaling"))
    lb = etree.Element(qn("c:logBase"))
    lb.set("val", "10")
    sc.insert(0, lb)
    xa.has_major_gridlines = False
    ya.has_major_gridlines = True
    ya.major_gridlines.format.line.color.rgb = FROST
    ya.major_gridlines.format.line.width = Pt(0.75)
    for ax in (xa, ya):
        style_axis_text(ax)
        ax.format.line.color.rgb = MIST
        ax.major_tick_mark = XL_TICK_MARK.NONE
    xa.tick_labels.number_format, xa.tick_labels.number_format_is_linked = "0.0", False
    ya.tick_labels.number_format, ya.tick_labels.number_format_is_linked = "0", False
    axis_title(xa, "Recall@10")
    axis_title(ya, "Milliseconds per query (log scale)")
    plot = ch.plots[0]
    for i, k in enumerate(order):
        ser = plot.series[i]
        st = ENG[k]
        ser.smooth = False
        ser.format.line.color.rgb = st["color"]
        ser.format.line.width = Pt(st["w"])
        if st["dash"]:
            ser.format.line.dash_style = st["dash"]
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 5
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = st["color"]
        ser.marker.format.line.color.rgb = st["color"]
        pt = ser.points[len(D[k]["results"]) - 1]
        dl = pt.data_label
        dl.has_text_frame = True
        dl.text_frame.text = st["name"]
        for r in dl.text_frame.paragraphs[0].runs:
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = st["color"]
            r.font.name = BODY
        dl.position = label_pos.get(k, XL_LABEL_POSITION.RIGHT)
    vb = plot.series[len(order)]
    vb.smooth = False
    vb.format.line.color.rgb = SLATE
    vb.format.line.width = Pt(1.0)
    vb.format.line.dash_style = MSO_LINE.SQUARE_DOT
    vb.marker.style = XL_MARKER_STYLE.NONE
    dl = vb.points[1].data_label
    dl.has_text_frame = True
    dl.text_frame.text = f"{int(vline * 100)}% recall"
    for r in dl.text_frame.paragraphs[0].runs:
        r.font.size = Pt(10)
        r.font.color.rgb = SLATE
        r.font.name = BODY
    dl.position = XL_LABEL_POSITION.LEFT
    return gf


def bar_chart(s, x, y, w, h, cats, vals, colors, fmt="0"):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("v", vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    tag(s, gf, "chart")
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = BODY
    ch.font.size = Pt(11)
    plot = ch.plots[0]
    plot.gap_width = 55
    plot.vary_by_categories = False
    ser = plot.series[0]
    for i, c in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(12)
    dl.font.bold = True
    dl.font.color.rgb = BLUECOAL
    dl.font.name = BODY
    ca, va = ch.category_axis, ch.value_axis
    ca.reverse_order = True
    ca.tick_labels.font.size = Pt(11)
    ca.tick_labels.font.color.rgb = BLUECOAL
    ca.format.line.color.rgb = MIST
    ca.major_tick_mark = XL_TICK_MARK.NONE
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale = 0
    va.maximum_scale = max(vals) * 1.22
    return gf


# =============================================================================== the deck ===
prs = deck()
D1, D2 = B.D1, B.D2

# ---------------------------------------------------------------------------- 1 title ---
s = title_slide(
    prs, "VECTOR SEARCH BENCHMARK REVIEW",
    statement="EDB Vector Plus, pgvector, VectorChord and ClickHouse on 1M and 2M rows of 768-dimension embeddings.",
    meta="Postgres 18.6 · WarehousePG 7.6 · ClickHouse 26.1 · 22 to 23 October 2026",
    byline="Source: six vp-bench-1 exports (runs 1 to 6) from the Vector Plus demo dashboard")
notes(s, "Six sweeps ran on 22 and 23 October: one at 1M rows and five at 2M. This deck covers what was tested "
         "and how, then the results, then what the sweeps could not show yet.")

# ------------------------------------------------------------------------- 2 summary ---
w_ivf, w_flat = B.lat_at(D2["w_ivfplus"], 0.90), B.lat_at(D2["w_ivfflat"], 0.90)
pg_ivf, pg_flat = B.lat_at(D2["ivfplus"], 0.85), B.lat_at(D2["ivfflat"], 0.85)
size_ivf, size_flat = D2["ivfplus"]["index_bytes"], D2["ivfflat"]["index_bytes"]
s = slide(prs)
header(s, "IVFPLUS IS FASTER AND HALF THE SIZE, BUT TOPS OUT AT 0.886 ON POSTGRES")
cw3 = row_widths(CW, 3, 0.5)
stats = [
    (f"{w_flat / w_ivf:.1f}×", f"faster than IVFFlat on WarehousePG at 90% recall: {ms(w_ivf)} against {ms(w_flat)}"),
    (f"{pg_flat / pg_ivf:.1f}×", f"faster than IVFFlat on Postgres at 85% recall: {ms(pg_ivf)} against {ms(pg_flat)}"),
    (f"{size_flat / size_ivf:.1f}×", f"smaller index than IVFFlat at 2M rows: {gib(size_ivf)} against {gib(size_flat)}"),
]
panel(s, LX - 0.15, BY + 0.05, cw3 + 0.3, 1.95, fill=DEEP)
for i, (big, cap) in enumerate(stats):
    x = LX + i * (cw3 + 0.5)
    if i == 0:
        stat(s, x + 0.12, BY + 0.2, cw3 - 0.24, big, cap, size=48, color=WHITE)
        s.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = MIST
    else:
        stat(s, x, BY + 0.2, cw3, big, cap, size=48)
for i in (1, 2):
    vrule(s, LX + i * (cw3 + 0.5) - 0.25, BY + 0.2, 1.7)
hrule(s, LX, 4.0, CW, color=MIST)
txt(s, LX, 4.22, 4.0, 0.32, "Still open", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
ceil = max(x["recall"] for x in D2["ivfplus"]["results"])
top_ms = max(x["ms"] for x in D2["ivfplus"]["results"])
open_items = [
    (f"ivfplus on Postgres plateaus at {ceil:.3f} recall.", f" Probes 64 to 256 add latency, up to {top_ms:.0f} ms, and no recall."),
    ("ClickHouse and ivfplus on WarehousePG were not extended.", " Their ladders end at 0.76 (ef_search 128) and 0.936 (32 probes)."),
    ("Runs 3 and 4 carry latency spikes.", " The 2M figures keep the quietest of four runs."),
]
for i, (lead, rest) in enumerate(open_items):
    txt_runs(s, LX, 4.78 + i * 0.58, CW, 0.5,
             [(lead, BODY, 16, True, BLUECOAL), (rest, BODY, 16, False, SLATE)], anchor=MID)
footer(s)
notes(s, "The speed comparison is at equal recall, read off the measured recall-latency curves. On WarehousePG both "
         "engines reach 90% recall, so that ratio is measured (6.1 ms against 19.7 ms at the points that first cross 90%, "
         "5.6 against 17.5 interpolated). On Postgres, ivfplus never reaches 90%: run 6 extended the ladder to 256 probes "
         "and its recall stays at 0.886 from 64 probes on. The comparison is therefore made at 85%, inside the range "
         "ivfplus covers. Index size is the on-disk index at 2M rows.")

# ----------------------------------------------------------------------- 3 divider ---
divider(prs, "THE TESTS", sub="Engines, data, queries and index builds")

# ------------------------------------------------------------------ 4 engines table ---
s = slide(prs)
header(s, "EACH ENGINE RAN ITS OWN INDEX ON ITS OWN COPY OF THE DATA")
cols = [("Engine", 2.6, L), ("Where it ran", 4.2, L), ("Index", 3.0, L), ("Build settings", 2.733, L)]
rows = [
    ["EDB Vector Plus", "Postgres 18.6, edb_vectorplus 0.0.5", "ivfplus (IVF with RaBitQ)", "lists 1414, rotation on"],
    ["EDB ivfplus @ WHPG", "WarehousePG 7.6, 2 segments, optimizer off", "ivfplus", "lists 1414"],
    ["pgvector IVFFlat", "Postgres 18.6, pgvector 0.8.6", "ivfflat", "lists 1414"],
    ["pgvector IVFFlat @ WHPG", "WarehousePG 7.6, pgvector 0.7.4", "ivfflat", "lists 1414"],
    ["pgvector HNSW", "Postgres 18.6, pgvector 0.8.6 (1M only)", "hnsw", "m 16, ef_construction 64"],
    ["VectorChord", "Postgres 18.6, vchord 1.1.1", "vchordrq", "lists [1414]"],
    ["ClickHouse", "26.1.2.1 EDB PoC preview build (2M only)", "vector_similarity (HNSW, bf16)", "M 16, ef_construction 128"],
]
rtable(s, LX, BY, 6.20, cols, rows, size=13)
txt(s, LX, 6.32, CW, 0.45,
    "Lists are 1000 at 1M rows and 1414 at 2M, the square root of the row count. Postgres builds ran with "
    "maintenance_work_mem 12 GB, no parallel workers and shared_buffers 128 MB.", size=12, color=SLATE)
footer(s)
notes(s, "Seven configurations. The two WarehousePG engines run on Postgres 12 with pgvector 0.7.4, an older stack than "
         "the Postgres 18 engines. HNSW was only built at 1M rows, and ClickHouse only at 2M.")

# ------------------------------------------------------------------- 5 data diagram ---
s = slide(prs)
header(s, "EVERY ENGINE HOLDS THE SAME 2,000,000 ROWS")
anchor_x, anchor_w = LX, 2.25
node(s, anchor_x, BY, anchor_w, 5.05, "vp_docs", kind="dark", size=18, sub="Postgres source table, also the ivfplus index")
cx, cwid = 3.55, 5.35
groups = [
    ("Postgres 18.6", ["pgi_docs|IVFFlat", "pgh_docs|HNSW", "vc_docs|VectorChord"]),
    ("WarehousePG 7.6", ["wvp_docs|ivfplus", "wpgi_docs|IVFFlat"]),
    ("ClickHouse 26.1", ["ch_docs|HNSW"]),
]
gh, ggap = 1.5, 0.275
for gi, (label, leaves) in enumerate(groups):
    gy = BY + gi * (gh + ggap)
    container(s, cx, gy, cwid, gh, label=label.upper())
    lw_ = row_widths(cwid - 0.4, 3, 0.15)
    for li, leaf in enumerate(leaves):
        name, sub = leaf.split("|")
        node(s, cx + 0.2 + li * (lw_ + 0.15), gy + 0.38, lw_, 0.85, name, kind="light", size=12, sub=sub)
    arrow(s, anchor_x + anchor_w, gy + gh / 2, cx, gy + gh / 2)
vrule(s, 9.25, BY + 0.1, 4.85)
tx, tw = 9.55, 3.38
txt(s, tx, BY + 0.1, tw, 0.3, "The rows", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
txt(s, tx, BY + 0.5, tw, 1.75,
    "Synthetic documents on ten topics. Each embedding is the normalized mean of random word vectors plus noise, "
    "768 dimensions.", size=13, color=SLATE)
txt(s, tx, BY + 2.65, tw, 0.3, "The check", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
txt(s, tx, BY + 3.05, tw, 2.0,
    "Before each comparison the app counts rows and compares the id range and an id checksum on every table. "
    "A mismatch stops the run.", size=13, color=SLATE)
footer(s)
notes(s, "Each table is a copy of vp_docs made by the dashboard: Postgres clones with COPY, WarehousePG tables by binary "
         "COPY, and ClickHouse by streaming the same rows. The fingerprint check is what makes recall comparable across "
         "engines: every engine is asked about the same rows.")

# -------------------------------------------------------------------- 6 protocol flow ---
s = slide(prs)
header(s, "EVERY SWEEP IS SCORED AGAINST AN EXACT SCAN OF THE SAME 50 QUERIES", size=22)
steps = [
    ("50 query vectors", "light", "Stored rows picked with a fixed seed, plus Gaussian noise of 0.05, then renormalized."),
    ("Exact top-10", "light", "A full sequential scan with index scans off. The ground truth for recall, computed once and reused."),
    ("Up to nine settings", "dark", "Probes 1, 2, 4 and so on to 32, extended to 256 in the last sweep. HNSW and ClickHouse get ef_search = max(k, 4 × probes)."),
    ("Three timed passes", "light", "One untimed warm-up pass first. Engines are interleaved, indexes prewarmed, JIT off."),
    ("Recall and latency", "light", "Recall@10 is the share of the exact top-10 returned. Latency is each query's median over the passes."),
]
gapx = 0.45
nw = row_widths(CW, 5, gapx)
for i, (label, kind, cap) in enumerate(steps):
    x = LX + i * (nw + gapx)
    node(s, x, 2.0, nw, 1.15, label, kind=kind, size=14)
    txt(s, x, 3.45, nw, 2.0, cap, size=14, color=SLATE, spacing=1.05)
    if i < 4:
        arrow(s, x + nw, 2.575, x + nw + gapx, 2.575)
hrule(s, LX, 5.3, CW, color=MIST)
txt(s, LX, 5.5, CW, 0.8,
    "Each result is the mean over the 50 queries. Recall carries a standard error of 0.01 to 0.05, and p95 is the "
    "95th percentile across queries.", size=14, color=BLUECOAL)
footer(s)
notes(s, "The 50 query vectors are real stored rows nudged with small noise, so each has genuine neighbours. The exact "
         "baseline took about 10.5 seconds per query at 2M rows and is computed once; later sweeps reuse it. For each "
         "setting the sweep runs one throwaway pass of three queries, then three timed passes, and keeps the median "
         "of the three for each query. Cold-cache mode exists but these sweeps were warm.")

# ------------------------------------------------------------------------ 7 the SQL ---
s = slide(prs)
header(s, "ONE QUERY SHAPE RUNS ON EVERY ENGINE", size=26)
lw_, rw_ = 6.45, CW - 6.45 - 0.33
rx_ = LX + lw_ + 0.33
txt(s, LX, BY, lw_, 0.3, "Postgres and WarehousePG", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
txt(s, rx_, BY, rw_, 0.3, "ClickHouse", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
code(s, LX, BY + 0.42, lw_, 2.55, [
    "SET ivfplus.probes = 16;",
    "-- ivfflat.probes = 16 | vchordrq.probes = '16'",
    "-- hnsw.ef_search = 64",
    "",
    "SELECT id, topic, category_id, title,",
    "       embedding <-> '[0.0412, -0.0157, …]' AS distance",
    "FROM public.vp_docs",
    "ORDER BY embedding <-> '[0.0412, -0.0157, …]'",
    "LIMIT 10;"], size=12)
code(s, rx_, BY + 0.42, rw_, 2.55, [
    "SELECT id, topic, category_id, title,",
    "       L2Distance(embedding, [0.0412, …])",
    "         AS distance",
    "FROM default.ch_docs",
    "ORDER BY L2Distance(embedding, [0.0412, …])",
    "LIMIT 10",
    "SETTINGS hnsw_candidate_list_size_for_search = 64;"], size=12)
txt(s, LX, 4.85, lw_, 1.6,
    ["Approximate runs set enable_seqscan = off and jit = off, so the planner cannot quietly scan the table.",
     "The exact baseline runs the same SELECT with enable_indexscan and enable_bitmapscan off."],
    size=14, color=SLATE, spacing=1.08)
txt(s, rx_, 4.85, rw_, 1.6,
    ["The vector is a literal of 768 numbers. Before each run the app checks the plan and records which index was used."],
    size=14, color=SLATE, spacing=1.08)
footer(s)
notes(s, "This is the exact statement text the dashboard logs in its Query Log tab. The query returns id, topic, category, "
         "title and distance, so every engine does the same fetch work. WarehousePG uses the same SELECT against its own "
         "tables with optimizer = off.")

# --------------------------------------------------------------------- 8 probes map ---
s = slide(prs)
header(s, "ONE LIST OF PROBES SETS THE EFFORT FOR EVERY ENGINE", size=26)
cols = [("Engine", 3.0, L), ("Setting", 4.1, L), ("Value for probes p", 2.5, L), ("Meaning", 2.93, L)]
rows = [
    ["EDB Vector Plus", "ivfplus.probes", "p, capped at lists", "lists scanned per query"],
    ["pgvector IVFFlat", "ivfflat.probes", "p, capped at lists", "lists scanned per query"],
    ["VectorChord", "vchordrq.probes", "'p', capped at lists", "lists scanned per query"],
    ["pgvector HNSW", "hnsw.ef_search", "max(k, 4 × p)", "candidate list size"],
    ["ClickHouse", "hnsw_candidate_list_size_for_search", "max(k, 4 × p)", "candidate list size"],
    ["WarehousePG twins", "same as their Postgres engine", "p, on each of 2 segments", "more total work per query"],
]
rtable(s, LX, BY, 5.55, cols, rows, size=13)
txt(s, LX, 5.75, CW, 0.9,
    "Runs 3 to 5 used p = 1 to 32, which gives HNSW and ClickHouse ef_search 10, 16, 32, 64, 128. Run 6 extended the ladder "
    "to 256 probes, where ef_search reaches its cap of 1000. The 1M sweep predates the shared list and used each engine's own list.",
    size=13, color=SLATE)
footer(s)
notes(s, "HNSW has no probes, so the shared value is translated into an ef_search that grows four times as fast, with "
         "a floor of k. That translation is a choice, not a standard: it decides where each curve starts, but the "
         "recall-latency charts compare at equal recall, which does not depend on it.")

# ---------------------------------------------------------------------- 9 index DDL ---
s = slide(prs)
header(s, "EACH INDEX WAS BUILT ONCE, WITH LISTS SET TO √ROWS", size=26)
code(s, LX, BY, CW, 3.3, [
    "CREATE INDEX … USING ivfplus   (embedding vector_l2_ops) WITH (lists = 1414, rotation = on);",
    "CREATE INDEX … USING ivfflat   (embedding vector_l2_ops) WITH (lists = 1414);",
    "CREATE INDEX … USING hnsw      (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);",
    "CREATE INDEX … USING vchordrq  (embedding vector_l2_ops)",
    "                WITH (options = $$ [build.internal]  lists = [1414] $$);",
    "",
    "ALTER TABLE ch_docs ADD INDEX vi embedding",
    "    TYPE vector_similarity('hnsw', 'L2Distance', 768, 'bf16', 16, 128)",
    "    GRANULARITY 100000000;",
    "ALTER TABLE ch_docs MATERIALIZE INDEX vi;"], size=13)
txt(s, LX, 5.2, CW, 1.3,
    ["Build time is one CREATE INDEX, or for ClickHouse ADD INDEX plus MATERIALIZE INDEX, measured once per engine.",
     "Postgres built serially (no parallel workers). WarehousePG builds on its 2 segments at the same time."],
    size=14, color=SLATE, spacing=1.1)
footer(s)
notes(s, "Lists are 1000 at 1M rows and 1414 at 2M. HNSW used pgvector's default ef_construction of 64, while ClickHouse "
         "used 128, its form default, so the two HNSW builds are not matched. Build times are single measurements.")

# --------------------------------------------------------------------- 10 divider ---
divider(prs, "THE RESULTS", sub="1M and 2M rows, recall against latency")

# ------------------------------------------------------------------ 11 six sweeps ---
s = slide(prs)
header(s, "SIX SWEEPS; THE 2M FIGURES USE THE QUIETEST OF FOUR RUNS", size=26)
cols = [("Run", 0.8, L), ("Saved", 1.9, L), ("Rows", 1.0, L), ("Engines", 1.2, L), ("What to know", 7.633, L)]
rows = [
    ["1", "22 Oct, 11:12", "1M", "6", "Includes HNSW; each engine used its own setting list; a single run."],
    ["2", "22 Oct, 16:09", "2M", "5", ("Set aside: the WarehousePG IVFFlat index was rebuilt afterwards (recall at 16 probes 0.928, then 0.916).", SLATE, False)],
    ["3", "23 Oct, 06:33", "2M", "6", "First sweep with ClickHouse; the exact baseline was computed here."],
    ["4", "23 Oct, 07:54", "2M", "6", "ClickHouse again; the baseline was reused."],
    ["5", "23 Oct, 08:19", "2M", "5", "No ClickHouse; the baseline was reused; the cleanest latency of runs 3 to 5."],
    ["6", "23 Oct, 10:28", "2M", "4", "Ladder extended to 256 probes; no ClickHouse or WarehousePG ivfplus; reproduces run 5."],
]
rtable(s, LX, BY, 5.65, cols, rows, size=13)
txt(s, LX, 5.85, CW, 0.85,
    "Recall is identical in runs 3 to 6, because the queries, the exact baseline and the indexes did not change. For each 2M "
    "setting the figure used here is the lowest latency of the runs that measured it; settings above 32 probes come from run 6 alone.",
    size=13, color=SLATE)
footer(s)
notes(s, "Run 2 predates the rebuild of the WarehousePG IVFFlat index at 16:26 on 22 October, so its WarehousePG IVFFlat "
         "recall differs from runs 3 to 6. It also carries a stale build record for that engine. Runs 3 to 6 share an exact "
         "baseline computed at 06:24 on 23 October. Run 6 recorded no ClickHouse note, and its latencies match run 5.")

# ---------------------------------------------------------------------- 12 1M chart ---
P = XL_LABEL_POSITION
s = slide(prs)
header(s, "AT 1M ROWS IVFPLUS ON WAREHOUSEPG REACHED 90% RECALL IN 4 MS", size=26)
order1 = ["w_ivfplus", "ivfplus", "w_ivfflat", "ivfflat", "vchordrq", "hnsw"]
recall_latency_chart(s, LX, BY, 8.9, 5.15, D1, order1,
                     {"w_ivfplus": P.ABOVE, "ivfplus": P.BELOW, "vchordrq": P.LEFT, "ivfflat": P.RIGHT, "w_ivfflat": P.RIGHT, "hnsw": P.LEFT})
vrule(s, 9.55, BY + 0.1, 4.95)
rx, rw = 9.85, 3.08
r1 = {r["key"]: r for r in B.score(D1, 0.90)}
panel(s, rx - 0.12, BY + 0.05, rw + 0.24, 1.75, fill=DEEP)
txt(s, rx, BY + 0.15, rw, 0.62, ms(r1["w_ivfplus"]["lat"]), size=36, font=DISPLAY_XB, color=WHITE, bold=True, anchor=MID)
txt(s, rx, BY + 0.80, rw, 0.95, f"ivfplus on WarehousePG at 91% recall, 16 probes", size=13, color=MIST)
stat(s, rx, BY + 2.05, rw, ms(r1["w_ivfflat"]["lat"]), "IVFFlat on WarehousePG at 95% recall", size=30, cap_h=0.5)
stat(s, rx, BY + 3.25, rw, ms(r1["hnsw"]["lat"]), "pgvector HNSW at 92% recall, ef_search 320", size=30, cap_h=0.5)
txt(s, rx, BY + 4.45, rw, 0.7, "ivfplus on Postgres reached 0.88 after 16 probes, the end of its list. At 2M it plateaus at 0.886.", size=13, color=SLATE)
footer(s)
notes(s, "At the 90% bar: ivfplus on WarehousePG 4.0 ms at probes 16 (recall 0.914), IVFFlat on WarehousePG 12.6 ms "
         "(0.946), pgvector HNSW 16.9 ms at ef_search 320 (0.920), VectorChord 33.6 ms (0.914), IVFFlat on Postgres "
         "34.8 ms (0.938). EDB Vector Plus on Postgres topped out at 0.882 because its 1M sweep list ended at 16 probes. "
         "This is a single run, so individual latencies carry noise; HNSW ran at 1M only.")

# ---------------------------------------------------------------------- 13 2M chart ---
s = slide(prs)
header(s, "AT 2M ROWS THE ORDER HOLDS: WAREHOUSEPG IVFPLUS STAYS UNDER 7 MS", size=26)
order2 = ["w_ivfplus", "ivfplus", "w_ivfflat", "ivfflat", "vchordrq", "clickhouse"]
recall_latency_chart(s, LX, BY, 8.9, 5.15, D2, order2,
                     {"ivfflat": P.ABOVE, "vchordrq": P.LEFT, "w_ivfflat": P.RIGHT, "ivfplus": P.LEFT, "w_ivfplus": P.RIGHT, "clickhouse": P.RIGHT})
vrule(s, 9.55, BY + 0.1, 4.95)
r2 = {r["key"]: r for r in B.score(D2, 0.90)}
panel(s, rx - 0.12, BY + 0.05, rw + 0.24, 1.75, fill=DEEP)
txt(s, rx, BY + 0.15, rw, 0.62, ms(r2["w_ivfplus"]["lat"]), size=36, font=DISPLAY_XB, color=WHITE, bold=True, anchor=MID)
txt(s, rx, BY + 0.80, rw, 0.95, "ivfplus on WarehousePG at 94% recall, 32 probes", size=13, color=MIST)
stat(s, rx, BY + 2.05, rw, ms(r2["w_ivfflat"]["lat"]), "IVFFlat on WarehousePG at 92% recall", size=30, cap_h=0.5)
stat(s, rx, BY + 3.25, rw, ms(r2["vchordrq"]["lat"]), "VectorChord at 90% recall; IVFFlat on Postgres took " + ms(r2["ivfflat"]["lat"]), size=30, cap_h=0.75)
txt(s, rx, BY + 4.5, rw, 0.65, "ivfplus on Postgres plateaus at 0.886 and ClickHouse stopped at 0.76; neither reached 90%.", size=13, color=SLATE)
footer(s)
notes(s, "2M figures are the lowest latency per setting across runs 3 to 6 (ClickHouse: runs 3 and 4). At the 90% bar "
         f"ivfplus on WarehousePG took {r2['w_ivfplus']['lat']:.1f} ms (recall 0.936, 32 probes), IVFFlat on WarehousePG {r2['w_ivfflat']['lat']:.1f} ms "
         f"(0.916, 16 probes), VectorChord {r2['vchordrq']['lat']:.1f} ms (0.904, 16 probes) and IVFFlat on Postgres {r2['ivfflat']['lat']:.1f} ms "
         "(0.906, 16 probes). EDB Vector Plus on Postgres plateaus at 0.886 from 64 probes to 256, so it never reaches the bar; "
         "ClickHouse stopped at 0.760 at ef_search 128, where its ladder ended.")

# ----------------------------------------------------------------- 14 equal recall ---
s = slide(prs)
lv = (0.60, 0.75, 0.85, 0.90, 0.95, 0.99)
_rt = lambda a, b, v: (B.lat_at(D2[b], v) / B.lat_at(D2[a], v)) if (B.lat_at(D2[a], v) and B.lat_at(D2[b], v)) else None
_all = [x for a, b in (("w_ivfplus", "w_ivfflat"), ("ivfplus", "ivfflat")) for v in lv for x in [_rt(a, b, v)] if x]
_wh = [x for v in lv for x in [_rt("w_ivfplus", "w_ivfflat", v)] if x]
_pg = [x for v in lv for x in [_rt("ivfplus", "ivfflat", v)] if x]
header(s, f"AT EQUAL RECALL IVFPLUS IS {min(_all):.1f} TO {max(_all):.1f}X FASTER THAN IVFFLAT", size=26)
LW, VW = 3.8, 1.4
cols = [("Latency at equal recall, 2M rows", LW, L)] + [(f"{int(round(v * 100))}%", VW, R) for v in lv] + [("", CW - LW - VW * 6, L)]
order_t = ["w_ivfplus", "w_ivfflat", "ivfplus", "ivfflat", "vchordrq", "clickhouse"]
rows = []
for k in order_t:
    row = [D2[k]["label"]]
    for v in lv:
        val = B.lat_at(D2[k], v)
        row.append(ms(val) if val else ("not reached", SLATE, False))
    row.append("")
    rows.append(row)
rh = rtable(s, LX, BY, 5.3, cols, rows, size=14)
ratio = lambda a, b, v: (B.lat_at(D2[b], v) / B.lat_at(D2[a], v)) if (B.lat_at(D2[a], v) and B.lat_at(D2[b], v)) else None
for i, (a, b, lab) in enumerate((("w_ivfplus", "w_ivfflat", "ivfplus vs IVFFlat on WarehousePG"),
                                  ("ivfplus", "ivfflat", "ivfplus vs IVFFlat on Postgres"))):
    y = 5.42 + i * 0.45
    txt(s, LX, y, LW - 0.1, 0.4, lab, size=13, bold=True, color=BLUECOAL, anchor=MID)
    for j, v in enumerate(lv):
        r_ = ratio(a, b, v)
        txt(s, LX + LW + j * VW, y, VW - 0.12, 0.4, f"{r_:.1f}×" if r_ else "–", size=14, bold=True,
            color=GREEN if r_ else SLATE, align=R, anchor=MID)
txt(s, LX, 6.36, CW, 0.4,
    "Interpolated along each measured curve, lowest of the repeats. ivfplus on Postgres plateaus at 0.886, so its comparison ends "
    "there; “not reached” means the engine never got that far.", size=11, color=SLATE)
footer(s)
notes(s, "Comparing at equal recall removes the effect of setting translation and of ivfplus needing more probes than "
         "IVFFlat for the same recall. On WarehousePG the advantage grows with recall, from " + f"{_wh[0]:.1f}× at 60% to {_wh[-1]:.1f}× at 90%; "
         "ivfplus on WarehousePG has no value at 95% or 99% because its ladder ended at 32 probes and 0.936. On Postgres "
         f"the advantage is {min(_pg):.1f}× to {max(_pg):.1f}× over the range ivfplus covers, which ends at 0.886: run 6 showed that more probes "
         "do not raise its recall. At 95% and 99% only the IVFFlat engines and, at 95%, VectorChord have values.")

# ----------------------------------------------------------------- 15 build and size ---
s = slide(prs)
header(s, "IVFPLUS INDEXES ARE HALF THE SIZE; BUILD TIME MATCHES IVFFLAT", size=26)
keys_b = sorted(D2, key=lambda k: D2[k]["build"]["build_seconds"])
keys_s = sorted(D2, key=lambda k: D2[k]["index_bytes"])
EDBC = lambda k: BLUECOAL if k in ("ivfplus", "w_ivfplus") else SLATE
half = row_widths(CW, 2, 0.5)
txt(s, LX, BY, half, 0.3, "Build time, seconds", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
bar_chart(s, LX, BY + 0.35, half, 3.9, [D2[k]["label"] for k in keys_b],
          [round(D2[k]["build"]["build_seconds"]) for k in keys_b], [EDBC(k) for k in keys_b], "0")
txt(s, LX + half + 0.5, BY, half, 0.3, "Index size, GiB", size=14, font=DISPLAY, color=BLUECOAL, bold=True)
bar_chart(s, LX + half + 0.5, BY + 0.35, half, 3.9, [D2[k]["label"] for k in keys_s],
          [round(D2[k]["index_bytes"] / B.GIB, 2) for k in keys_s], [EDBC(k) for k in keys_s], "0.00")
vrule(s, LX + half + 0.25, BY + 0.3, 3.9)
hrule(s, LX, 5.75, CW, color=MIST)
bpr = lambda k: D2[k]["index_bytes"] / 2_000_000
txt(s, LX, 5.92, CW, 0.8,
    f"Per row: ivfplus {bpr('ivfplus'):,.0f} bytes, IVFFlat {bpr('ivfflat'):,.0f}, VectorChord {bpr('vchordrq'):,.0f}, "
    f"ClickHouse {bpr('clickhouse'):,.0f}. A raw 768-dimension vector is 3,072 bytes. Builds are single measurements; "
    "the WarehousePG builds ran on 2 segments.", size=13, color=SLATE)
footer(s)
notes(s, "The ClickHouse figure is the secondary index only, the table data is separate. VectorChord and ClickHouse build "
         "fastest at about 470 s on one node; ivfplus and IVFFlat take about 680 s. On WarehousePG IVFFlat took "
         "406 s, the quickest build of all, but ivfplus took 650 s, so ivfplus gained little from the second segment.")

# --------------------------------------------------------------------- 16 scaling ---
s = slide(prs)
header(s, "DOUBLING THE ROWS ADDS 32 TO 56% LATENCY AND 2.5 TO 3.1X BUILD TIME", size=26)
cols = [("Engine", 3.0, L), ("Latency at 16 probes", 2.55, R), ("Recall at 16 probes", 2.35, R),
        ("Build time", 2.2, R), ("Index size", 2.43, R)]
keys_sc = ["ivfplus", "w_ivfplus", "ivfflat", "w_ivfflat", "vchordrq"]
rows = []
for k in keys_sc:
    a, b = B.at(D1, k, 16), B.at(D2, k, 16)
    ba, bb = D1[k]["build"]["build_seconds"], D2[k]["build"]["build_seconds"]
    sa, sb = D1[k]["index_bytes"], D2[k]["index_bytes"]
    rows.append([D2[k]["label"], f"{a['ms']:.1f} → {b['ms']:.1f} ms  (×{b['ms'] / a['ms']:.2f})",
                 f"{a['recall']:.3f} → {b['recall']:.3f}", f"×{bb / ba:.1f}", f"×{sb / sa:.1f}"])
rtable(s, LX, BY, 5.35, cols, rows, size=14)
txt(s, LX, 5.55, CW, 1.1,
    "Doubling the rows doubles the index and multiplies build time by 2.5 to 3.1. Query latency grows by less than "
    "the data, but recall at the same probes falls by 0.01 to 0.05, so a larger table needs a few more probes for the same recall. "
    "The 1M figures are a single run.", size=13, color=SLATE)
footer(s)
notes(s, "Compared at 16 probes. Lists changed from 1000 to 1414 with the row count, so each list holds about 1,400 rows at "
         "both sizes. The recall drop at fixed probes is the expected cost of a larger corpus. The 1M latencies come from one "
         "run and may carry noise, which can move these ratios by a few percent.")

# ----------------------------------------------------------------- 17 WHPG vs PG ---
s = slide(prs)
header(s, "WAREHOUSEPG CUTS IVFFLAT LATENCY 2.5X, BUT IVFPLUS MAINLY GAINS RECALL", size=26)
cols = [("Engine and probes", 3.4, L), ("Postgres", 2.9, R), ("WarehousePG", 2.9, R), ("Latency gain", 1.7, R), ("Recall gain", 1.633, R)]
rows = []
for eng, lab in (("ivfplus", "ivfplus"), ("ivfflat", "IVFFlat")):
    for p in (16, 32):
        a, b = B.at(D2, eng, p), B.at(D2, "w_" + eng, p)
        rows.append([f"{lab}, {p} probes", f"{a['ms']:.1f} ms · recall {a['recall']:.3f}",
                     f"{b['ms']:.1f} ms · recall {b['recall']:.3f}", f"{a['ms'] / b['ms']:.2f}×",
                     f"+{b['recall'] - a['recall']:.3f}"])
rtable(s, LX, BY, 4.9, cols, rows, size=14)
txt(s, LX, 5.15, 6.0, 1.5,
    "IVFFlat spends 50 ms or more per query on Postgres, so splitting its lists over two segments pays off. "
    "ivfplus already answers in 5 to 10 ms there, which leaves little for a second segment to save.", size=14, color=SLATE)
txt(s, 6.85, 5.15, 6.08, 1.5,
    "Probes apply on each segment, so the same setting searches more lists in total and returns higher recall. "
    "Compare at equal recall. WarehousePG also runs Postgres 12 with pgvector 0.7.4.", size=14, color=SLATE)
vrule(s, 6.667, 5.18, 1.25)
footer(s)
notes(s, "Same probes, same index definition, different clusters. For IVFFlat the latency gain is 2.5 to 2.7× and recall "
         "rises by 0.01. For ivfplus the latency gain is 1.1× at 16 probes and 1.7× at 32 probes, but recall rises by 0.04 to "
         "0.06 because each segment probes its own lists. The speed comparison on the equal-recall slide is the fair one.")

# ---------------------------------------------------------------- 18 ClickHouse ---
s = slide(prs)
header(s, "CLICKHOUSE HAS NOT REACHED 90% RECALL; ITS LATENCY IS FLAT AT 11 MS", size=26)
ch = D2["clickhouse"]["results"]
cd = CategoryChartData()
cd.categories = [str(r["param"]) for r in ch]
cd.add_series("ClickHouse recall", [r["recall"] for r in ch])
cd.add_series("90% recall bar", [0.9] * len(ch))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(LX), Inches(BY), Inches(8.3), Inches(4.7), cd)
tag(s, gf, "chart")
c = gf.chart
c.has_legend = False
c.has_title = False
c.font.name, c.font.size = BODY, Pt(11)
va, ca = c.value_axis, c.category_axis
va.minimum_scale, va.maximum_scale, va.major_unit = 0.0, 1.0, 0.2
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = FROST
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.0", False
for ax in (va, ca):
    style_axis_text(ax, 11)
    ax.format.line.color.rgb = MIST
    ax.major_tick_mark = XL_TICK_MARK.NONE
axis_title(ca, "ef_search (hnsw_candidate_list_size_for_search)", 11)
axis_title(va, "Recall@10", 11)
s0, s1 = c.plots[0].series
s0.smooth = s1.smooth = False
s0.format.line.color.rgb, s0.format.line.width = CORAL, Pt(3)
s0.marker.style, s0.marker.size = XL_MARKER_STYLE.CIRCLE, 8
s0.marker.format.fill.solid()
s0.marker.format.fill.fore_color.rgb = CORAL
s0.marker.format.line.color.rgb = CORAL
dl = s0.data_labels
dl.show_value = True
dl.number_format, dl.number_format_is_linked = "0.00", False
dl.position = XL_LABEL_POSITION.ABOVE
dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(12), True, CORAL
s1.format.line.color.rgb, s1.format.line.width = SLATE, Pt(1)
s1.format.line.dash_style = MSO_LINE.SQUARE_DOT
s1.marker.style = XL_MARKER_STYLE.NONE
d_ = s1.points[len(ch) - 1].data_label
d_.has_text_frame = True
d_.text_frame.text = "90% bar"
for r in d_.text_frame.paragraphs[0].runs:
    r.font.size, r.font.color.rgb, r.font.name = Pt(10), SLATE, BODY
d_.position = XL_LABEL_POSITION.ABOVE
vrule(s, 9.0, BY + 0.1, 4.5)
cx_, cw_ = 9.3, 3.63
stat(s, cx_, BY + 0.1, cw_, f"{max(r['recall'] for r in ch):.2f}", "best recall, at ef_search 128. pgvector HNSW at 1M rows needed ef_search 640 to reach 0.94.", size=34, cap_h=1.0)
stat(s, cx_, BY + 1.95, cw_, "11 ms", "at every setting from 10 to 128. A fixed per-query cost, the HTTP round trip included, seems to dominate.", size=34, cap_h=1.0)
stat(s, cx_, BY + 3.8, cw_, f"{D2['clickhouse']['build']['build_seconds']:.0f} s", f"to build, with a {gib(D2['clickhouse']['index_bytes'])} index (M 16, ef_construction 128, bf16).", size=34, cap_h=0.7)
txt(s, LX, 6.4, 8.3, 0.35, "Two latency spikes (68 and 76 ms) hit one setting in each run and did not repeat.", size=12, color=SLATE)
footer(s)
notes(s, "ClickHouse 26.1.2.1 EDB PoC preview build, HNSW with bf16 quantization, M 16, ef_construction 128. The recall "
         "curve was still rising at ef_search 128, so this is a floor and not a ceiling: 0.46, 0.51, 0.60, 0.70, 0.76. "
         "The flat 11 ms suggests the per-request cost (HTTP round trip, parsing, returning five columns) is larger than the "
         "search itself at these settings. The run 3 and 4 spikes were 76 ms at ef_search 64 and 68 ms at ef_search 32.")

# ------------------------------------------------------------------- 19 ranking ---
s = slide(prs)
header(s, "AT A 90% BAR WAREHOUSEPG IVFPLUS RANKS FIRST; AT 95% IVFFLAT TAKES OVER", size=22)
cols = [("Rank", 0.85, L), ("Engine", 3.15, L), ("Score", 1.05, R), ("Latency at bar", 1.85, R), ("p95", 1.35, R),
        ("Best recall", 1.5, R), ("Build", 1.3, R), ("Index", 1.483, R)]
sc = B.score(D2, 0.90)
q = [r for r in sc if r["qual"]]
best = lambda m: min(r[m] for r in q if r[m] is not None)
rows = []
n = 0
for r in sc:
    if r["qual"]:
        n += 1
    ok = r["qual"]
    rows.append([
        str(n) if ok else ("DNQ", SLATE, True),
        (r["label"], BLUECOAL if ok else SLATE, True),
        (f"{r['score']:.1f}" if ok else f"({r['score']:.1f})", BLUECOAL if ok else SLATE, ok),
        (ms(r["lat"]), GREEN, True) if (ok and r["lat"] == best("lat")) else (ms(r["lat"]) if ok else ("not reached", SLATE, False)),
        (ms(r["p95"]), GREEN, True) if (ok and r["p95"] == best("p95")) else (ms(r["p95"]) if ok else ""),
        f"{r['rec']:.3f}", f"{r['build']:.0f} s", f"{r['size'] / B.GIB:.2f} GiB"])
rtable(s, LX, BY, 5.4, cols, rows, size=14)
txt(s, LX, 5.62, CW, 1.05,
    ["Score: latency 40, p95 10, best recall 10, build 15, index size 15; maintenance was not measured. "
     "An engine that never reaches the bar is DNQ and ranks below every engine that does.",
     f"IVFFlat on WarehousePG leads at a 95% bar ({B.score(D2, 0.95)[0]['score']:.1f}) and at a 99% bar ({B.score(D2, 0.99)[0]['score']:.1f}). "
     "ivfplus on WarehousePG (best 0.936) is DNQ there because its ladder ended at 32 probes; at 99% only the two IVFFlat engines qualify."],
    size=13, color=SLATE, spacing=1.1)
footer(s)
notes(s, "This is the Benchmark tab's ranking recomputed on the consolidated 2M figures. The score is a summary, not a "
         "measurement: change the weights and the order can change. The 95% result for ivfplus on WarehousePG is a property "
         "of the sweep, not of the engine: its probe list stopped at 32, at 0.936. The DNQ for ivfplus on Postgres is different: "
         "run 6 showed its recall plateaus at 0.886, so it cannot reach 90% in this configuration. The Scoring tab in the "
         "dashboard shows the formula metric by metric.")

# ---------------------------------------------------------- 20 measurement quality ---
s = slide(prs)
header(s, "RUNS 3 AND 4 CARRY LATENCY SPIKES; RUN 6 REPRODUCES RUN 5", size=26)
cols = [("Same setting, four runs", 3.7, L), ("Run 3", 1.7, R), ("Run 4", 1.7, R), ("Run 5", 1.7, R), ("Run 6", 1.7, R), ("", 2.033, L)]


def quad(key, p):
    out = []
    for rid in (3, 4, 5, 6):
        e = B.RUNS[rid]["payload"]["engines"].get(key)
        out.append(next(z["ms"] for z in e["results"] if str(z["param"]) == str(p)) if e else None)
    return out


cases = [("EDB Vector Plus, 16 probes", "ivfplus", 16), ("VectorChord, 8 probes", "vchordrq", 8),
         ("pgvector IVFFlat, 32 probes", "ivfflat", 32), ("IVFFlat @ WHPG, 32 probes", "w_ivfflat", 32)]
rows = []
for lab, k, pr in cases:
    v = quad(k, pr)
    lo = min(v)
    rows.append([lab] + [(ms(x), SLATE, False) if x <= lo * 1.5 else (ms(x), BLUECOAL, True) for x in v] + [""])
rtable(s, LX, BY, 4.65, cols, rows, size=14)
b3, b4 = B.spikes(3), B.spikes(4)
ag = B.agreement(5, 6)
within = sum(abs(x - 1) <= 0.10 for x in ag)
med = sorted(ag)[len(ag) // 2]
txt(s, LX, 4.9, CW, 1.8,
    [f"Run 6 repeats run 5's settings from 1 to 32 probes: {within} of {len(ag)} latencies agree within 10%, the median ratio is "
     f"{med:.2f}, and recall is identical. Run 6 carries no warning that ClickHouse was running, and run 5 had no ClickHouse engine.",
     f"In runs 3 and 4, {b3[0]} and {b4[0]} of {b3[1]} settings ran more than 1.5 times slower than in run 5; both included "
     "ClickHouse on the same host. This deck keeps the lowest latency per setting."],
    size=14, color=SLATE, spacing=1.12)
footer(s)
notes(s, "Slowest outlier: VectorChord at 8 probes took 427 ms in run 3 and 186 ms in run 4 against 28 ms in run 5 and 28 ms in "
         "run 6. VectorChord is the most sensitive engine: at 64 and 128 probes in run 6 its pass-to-pass spread was large "
         "(standard deviation of the pass means above 500 ms), and those settings have no repeat. Run 6's note says the "
         "table check took 157 s, nearly all of it the WarehousePG IVFFlat table; running VACUUM ANALYZE on the WarehousePG "
         "tables is the likely fix, and it is a setup cost, not part of any measurement.")

# --------------------------------------------------------------------- 21 divider ---
divider(prs, "WHAT IS STILL OPEN", sub="Gaps in the data and the next sweep")

# ----------------------------------------------------------------------- 22 limits ---
s = slide(prs)
header(s, "TWO LADDERS STILL END EARLY: IVFPLUS ON WAREHOUSEPG AND CLICKHOUSE", size=24)
colw = row_widths(CW, 2, 0.5)
tab(s, LX, BY, "Not measured yet", w=colw)
tab(s, LX + colw + 0.5, BY, "Limits of what was measured", w=colw)
left = [("ivfplus on WarehousePG past 32 probes:", "recall was still rising at 0.936."),
        ("ClickHouse above ef_search 128:", "recall was still rising at 0.76."),
        ("Why ivfplus on Postgres stops at 0.886:", "the cause is not yet known."),
        ("pgvector HNSW at 2M rows, and churn:", "inserts, updates and deletes.")]
right = [("Settings above 32 probes:", "one run each, so no repeat to take the lowest of."),
         ("Synthetic embeddings, 50 queries:", "recall standard errors of 0.01 to 0.05."),
         ("ClickHouse shared the host", "with Postgres in runs 3 and 4."),
         ("WarehousePG runs Postgres 12:", "with pgvector 0.7.4, an older stack.")]
for col, items in ((0, left), (1, right)):
    x0 = LX + col * (colw + 0.5)
    for i, (lead, rest) in enumerate(items):
        y = BY + 0.62 + i * 1.12
        txt_runs(s, x0, y, colw, 0.9, [(lead + " ", BODY, 15, True, BLUECOAL), (rest, BODY, 15, False, SLATE)])
        if i < len(items) - 1:
            hrule(s, x0, y + 0.95, colw, color=MIST, lw=0.75)
vrule(s, LX + colw + 0.25, BY + 0.55, 4.3)
footer(s)
notes(s, "Run 6 closed one question: the Postgres ivfplus curve was not cut short, it is flat at 0.886 from 64 to 256 probes. "
         "Index maintenance under churn (10% of the ranking weight) was not run, so the scores are rescaled over the other "
         "five metrics. With shared_buffers at 128 MB, every index is served from the operating system cache.")

# ---------------------------------------------------------------- 23 next sweep ---
s = slide(prs)
header(s, "THE NEXT SWEEP SHOULD FINISH TWO LADDERS AND EXPLAIN ONE CEILING", size=24)
pw = row_widths(CW, 4, 0.4)
pillars = [
    ("01", "Finish the ladder", "Probes 64 to 256 for ivfplus on WarehousePG, and ef_search up to 1000 for ClickHouse."),
    ("02", "Explain the ceiling", "Rebuild ivfplus on Postgres with other lists, rotation and store_vectors settings, and see whether recall passes 0.9."),
    ("03", "Repeat on a quiet host", "Run each sweep three times with ClickHouse stopped, using the Repeats and quiet-host options now in the dashboard."),
    ("04", "Fill the gaps", "pgvector HNSW at 2M rows, a churn run for maintenance cost, then 10M rows."),
]
for i, (num, head, body) in enumerate(pillars):
    x = LX + i * (pw + 0.4)
    txt(s, x, BY + 0.7, pw, 0.9, num, size=54, font=DISPLAY_XB, color=BLUECOAL, bold=True, anchor=MID)
    hrule(s, x, BY + 1.75, pw, color=BLUECOAL, lw=2.0)
    txt(s, x, BY + 1.95, pw, 0.4, head, size=18, font=DISPLAY, color=BLUECOAL, bold=True)
    txt(s, x, BY + 2.5, pw, 2.4, body, size=15, color=SLATE, spacing=1.1)
footer(s)
notes(s, "The first item completes the two curves that still end early. The second matters most for the headline claim: "
         "ivfplus is faster at equal recall, but on Postgres it cannot be compared above 0.886 until the ceiling is explained. "
         "The third uses the dashboard's Repeats and quiet-host options, which save each repeat as its own run and let the "
         "Benchmark tab take the lowest or the median latency.")

# ------------------------------------------------------------------------- finish ---
mark_living_deck(prs, folder=OUT_DIR, build_script=os.path.basename(__file__))
apply_overrides(prs, DECK.replace(".pptx", ".overrides.json"))
save_deck(prs, DECK)
print("saved", DECK)
