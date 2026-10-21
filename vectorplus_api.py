#!/usr/bin/env python3
"""
EDB Vector Plus (edb_vectorplus) — Demo Dashboard API
Corpus generation, ivfplus index builds with timing, ANN search, recall
measurement, hybrid search and index maintenance — all driven from the UI.

Usage:
  python3 vectorplus_api.py [options]

Options:
  --pg-host HOST      PostgreSQL host        (default: localhost)
  --pg-port PORT      PostgreSQL port        (default: 5432)
  --pg-user USER      PostgreSQL user        (default: postgres)
  --pg-password PASS  PostgreSQL password    (default: )
  --pg-dbname DB      PostgreSQL database    (default: vectorplus_demo)
  --schema SCHEMA     Schema for demo objects(default: public)
  --app-port PORT     HTTP port for this API (default: 5066)
"""

import argparse
import io
import logging
import math
import os
import random
import re
import sqlite3
import struct
import threading
import time
import zlib
from datetime import datetime

import numpy as np
import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from pathlib import Path

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.environ.get("PGHOST",     "localhost"),
    "port":     int(os.environ.get("PGPORT", 5432)),
    "user":     os.environ.get("PGUSER",     "postgres"),
    "password": os.environ.get("PGPASSWORD", ""),
    "database": os.environ.get("PGDATABASE", "vectorplus_demo"),
}
SCHEMA     = os.environ.get("VP_SCHEMA", "public")
TABLE      = "vp_docs"
IDX_PREFIX = "vp_docs_emb"

# Binary COPY is roughly half the bytes of the text format and skips float
# formatting on the way out and float parsing on the way in. copy_rows() falls
# back to text automatically if a server refuses it, so this is safe to leave on.
COPY_BINARY_DEFAULT = os.environ.get("VP_COPY_BINARY", "1") not in ("0", "false", "no")

# Ceiling on the maintenance_work_mem the app will set for an index build on its
# own. A large lists value can ask for many GB; raising the session setting that
# far on a small box trades a clear error for an OOM kill.
BUILD_MAX_MAINT_MEM = os.environ.get("VP_BUILD_MAX_MAINT_MEM", "4GB")

# Facts straight out of the edb_vectorplus reference — surfaced in the UI so the
# demo explains the sharp edges instead of silently walking into them.
LIMITS = {
    "max_dim_vector":      2000,
    "max_dim_halfvec":     4000,
    "lists_min":           1,
    "lists_max":           32768,
    "probes_min":          1,
    "probes_max":          32768,
    "rerank_pool":         200,     # rows per query, fixed, no runtime knob
    "two_level_lists":     1800,    # lists >= this => 2-level, single-threaded build
    "packing_dim_max":     1780,    # at the default 8kB BLCKSZ
    "packing_dim_modulo":  4,
    "rotation_stages":     6,
    "epsilon0":            1.9,
    "fastscan_block":      32,
    "format_version":      1,
}

DESTRUCTIVE_SQL_RE = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|GRANT|REVOKE|CREATE|REINDEX|VACUUM)\b",
    re.IGNORECASE,
)

# pg_textsearch (Tiger Data) — BM25 as a Postgres index AM.
#   * needs shared_preload_libraries = 'pg_textsearch' and a restart
#   * PostgreSQL 17 and 18 only
#   * <@> returns the NEGATIVE BM25 score, because Postgres only supports
#     ascending index scans on an ordering operator. Lower is better, and
#     ORDER BY ... ASC is the relevant-first order.
#   * no term positions, so no phrase queries
BM25_EXPR   = "(coalesce(title, '') || ' ' || coalesce(body, ''))"
BM25_INDEX  = f"{TABLE}_bm25_idx"
BM25_DEFAULT_LIMIT = 1000     # pg_textsearch.default_limit, when no LIMIT given

OPCLASSES = {
    "vector_l2_ops":      {"type": "vector",  "op": "<->", "metric": "L2",            "default": True},
    "vector_ip_ops":      {"type": "vector",  "op": "<#>", "metric": "inner product", "default": False},
    "vector_cosine_ops":  {"type": "vector",  "op": "<=>", "metric": "cosine",        "default": False},
    "halfvec_l2_ops":     {"type": "halfvec", "op": "<->", "metric": "L2",            "default": False},
    "halfvec_ip_ops":     {"type": "halfvec", "op": "<#>", "metric": "inner product", "default": False},
    "halfvec_cosine_ops": {"type": "halfvec", "op": "<=>", "metric": "cosine",        "default": False},
}

# ── Runtime state ─────────────────────────────────────────────────────────────
GEN_STATE = {
    "running": False, "done": 0, "total": 0, "rows_per_sec": 0,
    "started": None, "elapsed": 0, "error": None, "phase": "idle", "dim": 0,
    "copy_seconds": None, "index_seconds": None,
    "dropped_indexes": [], "index_detail": [], "copy_format": None,
}
BUILD_STATE = {
    "running": False, "sql": "", "started": None, "elapsed": 0,
    "error": None, "notices": [], "phase": "idle", "pid": None,
    "progress": None, "result": None,
}
BUILD_HISTORY = []          # this session's builds; the durable record is in SQLite
CHURN_STATE = {"running": False, "phase": "idle", "done": 0, "total": 0, "error": None}


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT STORE  (shared with the vpbench CLI)
# ══════════════════════════════════════════════════════════════════════════════
# Builds and recall sweeps are written to a SQLite file rather than kept in
# memory, so numbers survive a restart. vpbench.py writes the same schema to the
# same file, which is how runs against a pure-pgvector database in vector_demo
# show up in this dashboard's comparison tab without the app having to connect
# to that database itself.

RESULTS_DB = os.environ.get("VP_RESULTS_DB", str(Path(__file__).resolve().parent / "vpbench.db"))
TARGET_NAME = os.environ.get("VP_TARGET", "edb")
TARGET_LABEL = os.environ.get("VP_TARGET_LABEL", "EDB Vector Plus")

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, target TEXT, label TEXT, engine TEXT, am TEXT, dsn_db TEXT,
    rows INTEGER, dim INTEGER,
    lists INTEGER, m INTEGER, ef_construction INTEGER,
    opclass TEXT, rotation INTEGER, concurrently INTEGER,
    index_name TEXT,
    build_seconds REAL, rows_per_sec REAL,
    index_bytes INTEGER, bytes_per_row REAL,
    maint_work_mem TEXT, mem_estimated_mb REAL, mem_required_mb REAL,
    parallel_workers TEXT, health REAL,
    pg_version TEXT, ext_version TEXT, notices TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS recalls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, build_id INTEGER, target TEXT, label TEXT, engine TEXT,
    index_name TEXT, rows INTEGER, dim INTEGER,
    k INTEGER, queries INTEGER,
    param TEXT, value INTEGER,
    recall REAL, ms REAL, p95_ms REAL, avg_returned REAL,
    exact_ms REAL, speedup REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, target TEXT, kind TEXT, label TEXT,
    sql TEXT, sql_full TEXT, settings TEXT, params TEXT,
    ms REAL, rows INTEGER, error TEXT
);
CREATE TABLE IF NOT EXISTS loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, target TEXT, rows INTEGER, dim INTEGER, seed INTEGER,
    seconds REAL, rows_per_sec REAL, generator TEXT
);
"""
_STORE_LOCK = threading.Lock()

# ── Query log ────────────────────────────────────────────────────────────────
# Every statement the app runs on the user's behalf goes through log_query, so
# the SQL is recoverable after the fact rather than only visible in whichever
# tab produced it. It lands in two places: a plain text file you can tail, and
# the results store so the UI can show and filter it.

LOG_FILE = os.environ.get("VP_LOG_FILE",
                          str(Path(__file__).resolve().parent / "vectorplus_queries.log"))
LOG_SQL_TO_CONSOLE = os.environ.get("VP_LOG_CONSOLE", "1") not in ("0", "false", "no")

# "user"  — every statement that touches vectors or full text, wherever it came
#           from: searches, hybrid legs, recall sampling, row previews, builds,
#           COPY, the SQL editor.
# "all"   — plus pure metadata: catalog lookups, current_setting, row counts,
#           pg_stat probes, health functions. The 4-second status poll is almost
#           entirely metadata, so this is off by default.
LOG_LEVEL = os.environ.get("VP_LOG_LEVEL", "user")

_qlog = logging.getLogger("vectorplus.queries")
_qlog.setLevel(logging.INFO)
_qlog.propagate = False


def _init_query_log():
    for h in list(_qlog.handlers):
        _qlog.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(fmt)
        _qlog.addHandler(fh)
    except Exception as e:
        print(f"  [log] cannot write {LOG_FILE}: {e}")
    if LOG_SQL_TO_CONSOLE:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        _qlog.addHandler(sh)


# Initialise on import as well as in __main__, so the log still works when the
# app is served by gunicorn or `flask run` rather than started directly.
_init_query_log()


# A 768-dimension vector literal is roughly 7 kB of text. Written in full to the
# log file it makes every other line unreadable, but it is also the one part of
# the query you cannot reconstruct afterwards. So: the full resolved statement
# always goes to the store, and the file gets an abbreviated form unless you ask
# otherwise with VP_LOG_FULL_SQL=1.
LOG_FULL_SQL_TO_FILE = os.environ.get("VP_LOG_FULL_SQL", "0") not in ("0", "false", "no")

_VEC_LITERAL_RE = re.compile(r"'\[[-0-9eE.,\s]{120,}\]'")


def resolve_sql(cur, sql, params):
    """The exact statement the server received, parameters substituted.

    psycopg2's mogrify does the same quoting the driver itself uses, so the
    result is runnable as-is in psql — which is the whole point of logging it.
    """
    if params is None:
        return None
    try:
        out = cur.mogrify(sql, params)
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out)
    except Exception:
        return None


def abbreviate_vectors(text, keep=4):
    """Shorten long vector literals for the file log, keeping the leading dims."""
    if not text:
        return text

    def _short(m):
        body = m.group(0)[2:-2]                       # strip quotes and brackets
        nums = [x for x in body.split(",") if x.strip()]
        head = ",".join(n.strip() for n in nums[:keep])
        return f"'[{head}, ... {len(nums)} dims]'"

    return _VEC_LITERAL_RE.sub(_short, text)


# Whether a statement is worth logging is a question about the statement, not
# about which function issued it. Anything touching the vector column or the
# full-text column is real work. Catalog lookups, current_setting, row counts
# and pg_stat probes are housekeeping, however they got here.
_DATA_SQL_RE = re.compile(
    r"\bembedding\b|\btsv\b|plainto_tsquery|to_tsvector|ts_rank|<->|<#>|<=>|halfvec",
    re.IGNORECASE)
_META_SQL_RE = re.compile(
    r"\bpg_catalog\b|\bpg_class\b|\bpg_index\b|\bpg_am\b|\bpg_namespace\b|"
    r"\bpg_attribute\b|\bpg_extension\b|\bpg_settings\b|\bpg_stat_|"
    r"\bpg_indexes\b|information_schema|current_setting|to_regclass|"
    r"pg_relation_size|pg_size_pretty|pg_total_relation_size|pg_backend_pid|"
    r"check_ivfplus_index_health|edb_vectorplus_rerank",
    re.IGNORECASE)


def classify_sql(kind, sql):
    """Retag a generic 'internal' statement based on what it actually does.

    Explicit kinds from the call sites (knn, hybrid-rrf, index-build, ...) are
    left alone; only the catch-all from q()/q1()/scalar() is reclassified.
    """
    if kind != "internal":
        return kind
    text = sql or ""
    if _DATA_SQL_RE.search(text) and not _META_SQL_RE.search(text):
        return "data"
    return "internal"


def log_query(kind, sql, settings=None, params=None, ms=None, rows=None,
              error=None, label=None, sql_full=None):
    """Record one statement. Never raise — logging must not break a working query."""
    kind = classify_sql(kind, sql)
    # 'internal' is metadata: catalogs, settings, row counts, health probes.
    # Suppressed unless you explicitly ask for everything.
    if kind == "internal" and LOG_LEVEL != "all":
        return
    try:
        sql_text = (sql or "").strip()
        setting_text = "; ".join(settings) if settings else ""
        head = f"[{kind}]"
        if ms is not None:
            head += f" {ms:.2f}ms"
        if rows is not None:
            head += f" {rows} rows"
        if error:
            head += " ERROR"
        # Prefer the resolved statement in the log line — it is the runnable one.
        line_src = sql_full or sql_text
        one_line = " ".join(line_src.split())
        if not LOG_FULL_SQL_TO_FILE:
            one_line = abbreviate_vectors(one_line)
        _qlog.info(f"{head}{(' {' + setting_text + '}') if setting_text else ''}  "
                   f"{one_line}" + (f"  -- {error}" if error else ""))
        store_insert("queries", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     target=TARGET_NAME, kind=kind, label=label, sql=sql_text,
                     sql_full=sql_full, settings=setting_text,
                     params=(str(params)[:500] if params is not None else None),
                     ms=round(ms, 3) if ms is not None else None,
                     rows=rows, error=(str(error)[:500] if error else None))
    except Exception:
        pass


def store_conn():
    c = sqlite3.connect(RESULTS_DB, timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript(STORE_SCHEMA)
    # Store files created before sql_full existed are still usable; add the
    # column rather than making the user delete their history.
    try:
        cols = {r[1] for r in c.execute("PRAGMA table_info(queries)")}
        if "sql_full" not in cols:
            c.execute("ALTER TABLE queries ADD COLUMN sql_full TEXT")
            c.commit()
    except Exception:
        pass
    return c


def store_insert(table, **kw):
    """Never let a bookkeeping failure take down a build that actually succeeded."""
    try:
        with _STORE_LOCK:
            c = store_conn()
            cols = ",".join(kw)
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(kw))})",
                            list(kw.values()))
            c.commit()
            rid = cur.lastrowid
            c.close()
            return rid
    except Exception as e:
        print(f"  [store] could not record {table}: {e}")
        return None


def human_bytes(nbytes):
    if nbytes is None:
        return "—"
    v, units, i = float(nbytes), ["B", "KiB", "MiB", "GiB", "TiB"], 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024; i += 1
    return f"{v:.{2 if v < 10 else 1}f} {units[i]}"


def store_query(sql, params=()):
    try:
        with _STORE_LOCK:
            c = store_conn()
            rows = [dict(r) for r in c.execute(sql, params).fetchall()]
            c.close()
            return rows
    except Exception as e:
        print(f"  [store] query failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC CORPUS  (text + embeddings that actually agree with each other)
# ══════════════════════════════════════════════════════════════════════════════
# Every document is drawn from one topic's vocabulary. The embedding is the
# normalized mean of per-word random vectors, so two documents about the same
# topic land near each other in vector space AND share words for full-text
# search. That is what makes the hybrid-search tab meaningful rather than
# decorative — the lexical and semantic rankings genuinely disagree at the
# margins, which is exactly when RRF earns its keep.

TOPICS = {
    "database-internals": {
        "category_id": 1,
        "words": """postgres heap tuple mvcc vacuum wal checkpoint buffer planner executor
                    btree index bloat autovacuum transaction isolation snapshot xid freeze
                    toast page relation catalog statistics analyze visibility""".split(),
    },
    "vector-search": {
        "category_id": 2,
        "words": """embedding vector nearest neighbour recall centroid quantization rabitq
                    ivfflat hnsw cosine euclidean probes clustering kmeans rerank ann
                    dimensionality latency index approximate distance semantic retrieval""".split(),
    },
    "kubernetes-ops": {
        "category_id": 3,
        "words": """kubernetes pod cluster node kubelet ingress deployment replica scheduler
                    namespace helm operator sidecar container runtime service mesh autoscaler
                    manifest rollout probe taint toleration""".split(),
    },
    "observability": {
        "category_id": 4,
        "words": """metrics tracing logging prometheus grafana span histogram cardinality
                    alerting dashboard latency percentile telemetry exporter scrape sampling
                    incident oncall runbook anomaly threshold""".split(),
    },
    "security": {
        "category_id": 5,
        "words": """encryption tls certificate rotation credential secret vault authentication
                    authorization audit compliance breach firewall injection privilege
                    hardening cipher token session revocation""".split(),
    },
    "coffee": {
        "category_id": 6,
        "words": """espresso roast grinder brewing arabica portafilter crema tamper burr pour
                    filter extraction bean aroma acidity milk latte barista dose yield""".split(),
    },
    "cycling": {
        "category_id": 7,
        "words": """bicycle cadence peloton gearing derailleur climbing descent saddle wheelset
                    tubeless watts threshold sprint gravel commuter frame carbon puncture""".split(),
    },
    "cooking": {
        "category_id": 8,
        "words": """braise simmer sear stock reduction seasoning knife julienne emulsion dough
                    proofing ferment umami marinade roasting skillet garnish plating""".split(),
    },
    "finance": {
        "category_id": 9,
        "words": """portfolio yield equity volatility hedging derivative liquidity ledger
                    settlement arbitrage dividend valuation risk exposure benchmark drawdown
                    custody clearing spread""".split(),
    },
    "climate": {
        "category_id": 10,
        "words": """emissions carbon warming glacier drought monsoon albedo sequestration
                    renewable turbine photovoltaic mitigation adaptation permafrost sea level
                    aerosol biodiversity""".split(),
    },
}
TOPIC_NAMES = list(TOPICS.keys())
FILLER = """the a of and to in for with on that this these those from is are was were
            using when after before during over under between about each per""".split()

TITLE_FORMS = [
    "Notes on {a} and {b}",
    "Why {a} matters for {b}",
    "Tuning {a}: a practical guide",
    "{a} versus {b} in production",
    "Debugging {a} problems",
    "A short field guide to {a}",
    "What we learned about {a}",
    "{a}, {b}, and everything after",
]

# Norm of the per-document jitter added on top of the unit-norm topic signal.
# ~0.25 keeps documents distinct without dissolving the topic clusters that the
# ivfplus k-means build needs in order to learn useful centroids.
DOC_JITTER = 0.25


# ── Banking corpus ───────────────────────────────────────────────────────────
# Selected with --corpus banking (or VP_CORPUS=banking). The topics map onto the
# retrieval workloads in the Mizuho briefing: policy and procedure, KYC/AML,
# credit and market research, trade and contract documentation.
#
# Note this is still synthetic text. It changes the *vocabulary* so the demo
# reads as a bank's corpus, not the statistics — document lengths remain uniform
# and IDF remains flat, so it is no better than the technology corpus for a
# ranking-quality claim. Only real documents with relevance judgements can
# support that.
TOPICS_BANKING = {
    "policy-procedure": {"category_id": 1, "words": """policy procedure control framework
        mandate approval delegation authority escalation exception threshold attestation
        governance committee review standard directive circular compliance manual""".split()},
    "regulatory-filing": {"category_id": 2, "words": """regulator supervisor filing disclosure
        capital liquidity ratio basel reporting submission remediation finding examination
        consent order deadline jurisdiction licence prudential returns""".split()},
    "kyc-aml": {"category_id": 3, "words": """onboarding beneficial ownership screening sanctions
        watchlist adverse media politically exposed identity verification transaction
        monitoring alert disposition suspicious typology remediation refresh risk rating""".split()},
    "credit-risk": {"category_id": 4, "words": """obligor exposure limit rating downgrade default
        probability collateral covenant facility drawdown provision impairment stress
        concentration counterparty recovery watchlist deterioration""".split()},
    "market-research": {"category_id": 5, "words": """issuer sector outlook valuation yield spread
        duration curve inflation policy rate earnings guidance consensus target
        overweight underweight catalyst estimate revision""".split()},
    "trade-finance": {"category_id": 6, "words": """letter credit guarantee documentary collection
        bill lading shipment beneficiary applicant confirming advising discrepancy
        presentation tenor negotiation reimbursement incoterm""".split()},
    "derivatives-legal": {"category_id": 7, "words": """isda master agreement schedule annex
        collateral margin threshold netting termination event default close out valuation
        confirmation novation clearing counterparty settlement""".split()},
    "payments-ops": {"category_id": 8, "words": """settlement clearing nostro reconciliation
        instruction swift message correspondent cutoff value date repair exception break
        investigation recall return liquidity intraday""".split()},
    "operational-risk": {"category_id": 9, "words": """incident outage root cause remediation
        control failure loss event near miss reporting escalation resilience continuity
        recovery objective dependency vendor concentration""".split()},
    "customer-service": {"category_id": 10, "words": """enquiry complaint resolution branch
        channel account statement dispute chargeback fee waiver retention feedback
        escalation satisfaction turnaround acknowledgement""".split()},
}
BANKING_TITLE_FORMS = [
    "Standard on {a} and {b}",
    "{a} review: findings and actions",
    "Guidance note — {a}",
    "{a} versus {b}: operating practice",
    "Escalation procedure for {a}",
    "Annual assessment of {a}",
    "Control narrative: {a} and {b}",
    "{a} exception handling",
]


# corpus_bfsi composes whole documents rather than sampling a word bag, so it
# takes over generation entirely rather than just swapping the vocabulary.
CORPUS_COMPOSE = None


def select_corpus(name):
    """Swap the corpus the generator draws from. Returns the corpus name."""
    global TOPICS, TOPIC_NAMES, TITLE_FORMS, CORPUS_NAME, CORPUS_COMPOSE
    CORPUS_COMPOSE = None
    if name == "banking":
        TOPICS = TOPICS_BANKING
        TITLE_FORMS = BANKING_TITLE_FORMS
    elif name == "bfsi":
        try:
            import corpus_bfsi
        except ImportError as e:
            raise ValueError("corpus_bfsi.py must sit next to this file "
                             f"to use --corpus bfsi ({e})")
        TOPICS = corpus_bfsi.TOPICS
        CORPUS_COMPOSE = corpus_bfsi.compose
    elif name != "tech":
        raise ValueError(f"Unknown corpus {name!r} — use tech, banking or bfsi")
    TOPIC_NAMES = list(TOPICS)
    CORPUS_NAME = name
    return name


CORPUS_NAME = "tech"


_WORD_VEC_CACHE = {}


def word_vec(word: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-embedding for one token. Cached per (word, dim)."""
    key = (word, dim)
    hit = _WORD_VEC_CACHE.get(key)
    if hit is None:
        seed = zlib.crc32(word.encode()) ^ (dim * 2654435761 & 0xFFFFFFFF)
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        hit = rng.standard_normal(dim).astype(np.float32)
        _WORD_VEC_CACHE[key] = hit
    return hit


_TOKEN_RE = re.compile(r"[^a-zA-Z0-9]+")


def embed_text(text: str, dim: int) -> np.ndarray:
    """Mean of token vectors, L2-normalized. Same function for docs and queries."""
    toks = [t for t in _TOKEN_RE.split(text.lower()) if len(t) > 2]
    if not toks:
        return np.zeros(dim, dtype=np.float32)
    acc = np.zeros(dim, dtype=np.float32)
    for t in toks:
        acc += word_vec(t, dim)
    n = np.linalg.norm(acc)
    return acc / n if n > 0 else acc


_VEC_TPL = {}


def vec_literal(v: np.ndarray) -> str:
    """pgvector text form: [0.1,0.2,...] — 5 decimals keeps COPY payloads sane.

    One %-format against a pre-built template is ~5x faster than np.char.mod()
    and produces byte-identical output. At 768 dims this was the single largest
    cost in the loader, ahead of generating the document itself.
    """
    n = v.shape[0]
    tpl = _VEC_TPL.get(n)
    if tpl is None:
        tpl = "[" + ",".join(["%.5f"] * n) + "]"
        _VEC_TPL[n] = tpl
    return tpl % tuple(v.tolist())


# ── Binary COPY encoding ──────────────────────────────────────────────────────
# Text COPY spends most of its time turning floats into decimal strings and then
# making the server parse them back. The binary format skips both: the payload
# is roughly half the size and encoding is a memory copy. Columns, in order:
# topic (text), category_id (int4), title (text), body (text), embedding (vector).
# pgvector's binary layout is int16 dim, int16 unused, then dim x float4, all
# big-endian.

_COPY_BIN_HEADER  = b"PGCOPY\n\xff\r\n\x00" + struct.pack("!ii", 0, 0)
_COPY_BIN_TRAILER = struct.pack("!h", -1)
_HDR_5F  = struct.pack("!h", 5)
_INT4LEN = struct.pack("!i", 4)


def _bin_row(topic: str, cat: int, title: str, body: str, emb: np.ndarray) -> bytes:
    tb = topic.encode("utf-8")
    ti = title.encode("utf-8")
    bo = body.encode("utf-8")
    vec = struct.pack("!hh", emb.shape[0], 0) + emb.astype(">f4").tobytes()
    return b"".join((
        _HDR_5F,
        struct.pack("!i", len(tb)), tb,
        _INT4LEN, struct.pack("!i", int(cat)),
        struct.pack("!i", len(ti)), ti,
        struct.pack("!i", len(bo)), bo,
        struct.pack("!i", len(vec)), vec,
    ))


def copy_buffer_binary(rows) -> io.BytesIO:
    """Whole batch as one binary COPY stream."""
    parts = [_COPY_BIN_HEADER]
    parts.extend(_bin_row(*r) for r in rows)
    parts.append(_COPY_BIN_TRAILER)
    buf = io.BytesIO(b"".join(parts))
    buf.seek(0)
    return buf


def copy_buffer_text(rows) -> io.StringIO:
    """Whole batch as one text COPY stream. Joining once beats repeated write()."""
    out = []
    for topic, cat, title, body, emb in rows:
        t = title.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        b = body.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
        out.append(f"{topic}\t{cat}\t{t}\t{b}\t{vec_literal(emb)}\n")
    buf = io.StringIO("".join(out))
    buf.seek(0)
    return buf


COPY_COLS = "(topic, category_id, title, body, embedding)"


def copy_rows(cur, rows, binary=True, log_kind="copy"):
    """COPY one batch in. Returns the format actually used, so a server that
    refuses binary degrades to text instead of failing the load."""
    _t = time.perf_counter()
    if binary:
        try:
            cur.copy_expert(
                f"COPY {qualified(TABLE)} {COPY_COLS} FROM STDIN WITH (FORMAT BINARY)",
                copy_buffer_binary(rows))
            log_query(log_kind,
                      f"COPY {qualified(TABLE)} {COPY_COLS} FROM STDIN WITH (FORMAT BINARY)",
                      ms=(time.perf_counter() - _t) * 1000, rows=len(rows))
            return "binary"
        except Exception:
            # The failed COPY may have left the connection mid-transaction, so
            # clear it before retrying the same rows as text.
            try:
                cur.connection.rollback()
            except Exception:
                pass
    cur.copy_expert(f"COPY {qualified(TABLE)} {COPY_COLS} FROM STDIN",
                    copy_buffer_text(rows))
    log_query(log_kind, f"COPY {qualified(TABLE)} {COPY_COLS} FROM STDIN",
              ms=(time.perf_counter() - _t) * 1000, rows=len(rows))
    return "text"


def make_document(rng: np.random.Generator, dim: int):
    """One synthetic doc: topic, category, title, body, embedding.

    Kept for single-document callers; the loader uses make_batch().
    """
    return make_batch(rng, dim, 1, pyrng=random.Random(int(rng.integers(1 << 31))))[0]


def make_batch(rng: np.random.Generator, dim: int, n: int, pyrng=None):
    """n synthetic documents at once.

    Same distributions as the per-document version, but the expensive numpy work
    is hoisted out of the loop: one noise matrix for the whole batch instead of
    n calls, and word sampling moved to the stdlib RNG, which is ~5x faster than
    numpy's choice() on a list of strings.
    """
    pyrng = pyrng or random
    # A raw standard normal has norm ~sqrt(dim), which would swamp the unit-norm
    # signal and leave the data near-isotropic — and near-isotropic data makes
    # k-means centroids meaningless, which would flatten the recall curve and
    # misrepresent the index. Scale the jitter relative to sqrt(dim).
    noise = rng.standard_normal((n, dim), dtype=np.float32) * (DOC_JITTER / math.sqrt(dim))

    n_topics = len(TOPIC_NAMES)
    n_forms  = len(TITLE_FORMS)
    out = []

    if CORPUS_COMPOSE is not None:
        for i in range(n):
            topic, cat, title, body = CORPUS_COMPOSE(pyrng)
            emb = embed_text(title + " " + body, dim) + noise[i]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            out.append((topic, cat, title, body, emb))
        return out

    for i in range(n):
        topic = TOPIC_NAMES[pyrng.randrange(n_topics)]
        spec  = TOPICS[topic]
        vocab = spec["words"]
        picks = pyrng.sample(vocab, min(14, len(vocab)))
        fill  = pyrng.choices(FILLER, k=10)

        title = TITLE_FORMS[pyrng.randrange(n_forms)].format(a=picks[0], b=picks[1])

        body_words = []
        for j, w in enumerate(picks):
            body_words.append(w)
            if j % 2 == 1:
                body_words.append(fill[j % len(fill)])
        body = " ".join(body_words).capitalize() + "."

        emb = embed_text(title + " " + body, dim) + noise[i]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        out.append((topic, spec["category_id"], title, body, emb))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_conn(autocommit=False):
    c = psycopg2.connect(**DB_CONFIG, connect_timeout=8)
    c.autocommit = autocommit
    return c


def qualified(name: str) -> str:
    return f'"{SCHEMA}"."{name}"'


def q(sql, params=None, timeout_ms=30000, kind="internal"):
    """Every read the app makes goes through here, so this is where the
    housekeeping statements get logged. q1() and scalar() both delegate to it,
    so instrumenting this one function covers all three without double-counting."""
    conn = get_conn()
    t0 = time.perf_counter()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
        try:
            cur.execute(sql, params)
        except Exception as e:
            log_query(kind, sql, params=params, error=e,
                      sql_full=resolve_sql(cur, sql, params))
            raise
        rows = cur.fetchall() if cur.description else []
        conn.commit()
        log_query(kind, sql, params=params,
                  ms=(time.perf_counter() - t0) * 1000.0, rows=len(rows),
                  sql_full=resolve_sql(cur, sql, params))
        return [dict(r) for r in rows]
    finally:
        conn.close()


def q1(sql, params=None, timeout_ms=30000, kind="internal"):
    r = q(sql, params, timeout_ms, kind=kind)
    return r[0] if r else None


def scalar(sql, params=None, default=None, timeout_ms=30000, kind="internal"):
    try:
        r = q1(sql, params, timeout_ms, kind=kind)
        if not r:
            return default
        return list(r.values())[0]
    except Exception:
        return default


def terms_absent_from_corpus(text, limit=6):
    """Which query words occur nowhere in the table — the usual reason a
    lexical search returns nothing after switching corpus."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if len(w) > 2]
    if not words:
        return []
    try:
        rows = q(f"""SELECT w FROM unnest(%s::text[]) AS w
                     WHERE NOT EXISTS (
                        SELECT 1 FROM {qualified(TABLE)}
                        WHERE tsv @@ plainto_tsquery('english', w) LIMIT 1)""",
                 (words,), timeout_ms=15000)
        return [r["w"] for r in rows][:limit]
    except Exception:
        return []


def bm25_available():
    """Is pg_textsearch installed, loaded and usable in this database?"""
    out = {"extension": None, "am": False, "preloaded": False, "index": None,
           "usable": False, "reason": None}
    out["extension"] = scalar(
        "SELECT extversion FROM pg_extension WHERE extname = 'pg_textsearch'")
    out["am"] = bool(scalar("SELECT 1 FROM pg_am WHERE amname = 'bm25'", default=0))
    pre = scalar("SELECT current_setting('shared_preload_libraries')") or ""
    out["preloaded"] = "pg_textsearch" in pre
    out["pg_major"] = int((scalar("SELECT current_setting('server_version_num')")
                           or 0)) // 10000
    try:
        idx = q("""SELECT i.relname AS name, pg_relation_size(i.oid) AS bytes,
                          pg_get_indexdef(i.oid) AS indexdef
                   FROM pg_class i JOIN pg_am am ON am.oid = i.relam
                   JOIN pg_index x ON x.indexrelid = i.oid
                   JOIN pg_class t ON t.oid = x.indrelid
                   WHERE am.amname = 'bm25' AND t.relname = %s""", (TABLE,))
    except Exception:
        # Same degradation as every other status endpoint: report the problem
        # rather than 500 when the database is unreachable.
        idx = []
    out["index"] = idx[0] if idx else None
    if out["index"]:
        out["index"]["size"] = human_bytes(out["index"]["bytes"])

    if not out["extension"]:
        out["reason"] = ("pg_textsearch is not installed in this database. "
                         "Run CREATE EXTENSION pg_textsearch.")
    elif not out["am"]:
        out["reason"] = "The bm25 access method is missing — the library did not load."
    elif not out["index"]:
        out["reason"] = "No BM25 index on the demo table yet — build one."
    else:
        out["usable"] = True
    if out["pg_major"] and out["pg_major"] < 17:
        out["reason"] = (f"pg_textsearch supports PostgreSQL 17 and 18; this server "
                         f"is {out['pg_major']}.")
        out["usable"] = False
    return out


def clean_notices(raw):
    out = []
    for n in raw:
        s = str(n).strip()
        if s:
            out.append(s)
    return out[-20:]


def table_dim():
    """Current declared dimension of the embedding column, or 0 if no table."""
    v = scalar("""
        SELECT a.atttypmod
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding'
    """, (SCHEMA, TABLE), default=0)
    return int(v) if v and v > 0 else 0


def ivfplus_indexes():
    """Every ivfplus index on the demo table, with size and build options."""
    try:
        rows = q("""
            SELECT i.relname                              AS name,
                   pg_relation_size(i.oid)                AS bytes,
                   pg_size_pretty(pg_relation_size(i.oid)) AS size,
                   am.amname                              AS am,
                   pg_get_indexdef(i.oid)                 AS indexdef,
                   x.indisvalid                           AS valid,
                   x.indpred IS NOT NULL                  AS partial
            FROM pg_class i
            JOIN pg_index x    ON x.indexrelid = i.oid
            JOIN pg_class t    ON t.oid = x.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_am am      ON am.oid = i.relam
            WHERE n.nspname = %s AND t.relname = %s
              AND am.amname IN ('ivfplus','ivfflat','hnsw')
            ORDER BY i.relname
        """, (SCHEMA, TABLE))
    except Exception:
        return []
    for r in rows:
        m = re.search(r"lists\s*=\s*'?(\d+)", r.get("indexdef") or "")
        r["lists"] = int(m.group(1)) if m else None
        m2 = re.search(r"USING\s+\w+\s*\(\s*\w+\s+(\w+)", r.get("indexdef") or "")
        r["opclass"] = m2.group(1) if m2 else None
    return rows


def index_health(name: str):
    """check_ivfplus_index_health() — 6 columns, plus whatever NOTICE it raised."""
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM check_ivfplus_index_health(%s::regclass)",
                    (f"{SCHEMA}.{name}",))
        row = cur.fetchone()
        notices = clean_notices(conn.notices)
        conn.commit()
        out = dict(row) if row else {}
        out["notices"] = notices
        return out
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def packing_window(dim: int, blcksz: int = 8192):
    """
    The one limit with an action attached: a vector index only gets the packed
    fast-scan layout when dim %% 4 == 0 and the packed block fits one page
    (dim <= 1780 at the default 8kB BLCKSZ). Outside it the index silently runs
    the slower unpacked path and health reports 1.0 forever.
    """
    dim_max = LIMITS["packing_dim_max"] if blcksz == 8192 else int(blcksz / 8 * 1.74)
    mod_ok  = dim % LIMITS["packing_dim_modulo"] == 0
    fits    = dim <= dim_max
    reasons = []
    if not mod_ok:
        reasons.append(f"dim {dim} is not a multiple of {LIMITS['packing_dim_modulo']}")
    if not fits:
        reasons.append(f"dim {dim} exceeds {dim_max} for BLCKSZ {blcksz}")
    return {
        "packed": mod_ok and fits,
        "dim": dim, "dim_max": dim_max, "blcksz": blcksz,
        "reasons": reasons,
        "suggestion": (dim // 4) * 4 if not mod_ok else (dim_max if not fits else dim),
    }


def parse_mem_setting(s):
    """PostgreSQL memory setting -> bytes. A bare number means kB, as in postgresql.conf."""
    if s is None:
        return None
    m = re.match(r"^\s*(\d+)\s*(B|kB|MB|GB|TB)?\s*$", str(s), re.IGNORECASE)
    if not m:
        return None
    mult = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4}
    return int(m.group(1)) * mult[(m.group(2) or "kB").lower()]


def fmt_mb(nbytes):
    return f"{nbytes / 1024 / 1024:.0f}MB"


# The server states the real figure in its failure message. We parse it and
# retry at that value, so the model below only has to be close enough to avoid a
# pointless first attempt — it never has to be exactly right.
MEM_ERROR_RE = re.compile(
    r"memory required is (\d+)\s*MB,\s*maintenance_work_mem is (\d+)\s*MB", re.I)


def _maxalign(n):
    return (n + 7) & ~7


def ivf_build_memory(rows: int, dim: int, lists: int):
    """Working set of an IVF k-means build, in bytes, with the terms broken out.

    An earlier version of this counted only the sample array and the centroids
    and undershot by more than half. The term it missed is Elkan's lower-bound
    matrix: one float per (sample, centre) pair. Since the sample count is
    50 x lists, that matrix is 50 x lists^2 floats, which is what the reference
    means when it says build memory rises quadratically with lists.

    Checked against two real server errors at dim 768:

        lists = 1000   model 350.6 MB   server "memory required is 351 MB"
        lists = 1780   model 894.2 MB   server "memory required is 895 MB"

    Still only an estimate. Past lists >= 1800 ivfplus builds the hierarchy
    top-down, so each k-means run is smaller and this becomes an upper bound
    rather than a prediction.
    """
    samples = max(50 * lists, 10000)
    if rows:
        samples = min(samples, rows)
    itemsize = _maxalign(8 + dim * 4)            # Vector: header + dim floats

    parts = {
        "sample_array":  samples * itemsize,
        "lower_bound":   samples * lists * 4,    # Elkan — the dominant term
        "centers":       3 * lists * itemsize,   # centers, newCenters, aggregate
        "half_distance": lists * lists * 4,      # Elkan — O(lists^2)
        "bookkeeping":   samples * 4 + samples * 4 + lists * 8,
    }
    total = sum(parts.values())
    return {"samples": samples, "bytes": total,
            "mb": round(total / 1024 / 1024, 1), "pretty": fmt_mb(total),
            "parts": parts, "largest_term": max(parts, key=parts.get),
            "largest_term_mb": round(max(parts.values()) / 1024 / 1024, 1)}


def max_lists_for_memory(mem_bytes: int, dim: int, rows: int):
    """Largest lists value whose k-means sample still fits in mem_bytes."""
    if not dim or not mem_bytes:
        return LIMITS["lists_min"]
    lo, hi, best = LIMITS["lists_min"], LIMITS["lists_max"], LIMITS["lists_min"]
    while lo <= hi:
        mid = (lo + hi) // 2
        if ivf_build_memory(rows, dim, mid)["bytes"] <= mem_bytes:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return best


def recommend_lists(rows: int) -> dict:
    """lists is NOT derived from table size — the default of 100 stays 100 at a
    billion rows. This is the doc's sizing table, made clickable."""
    if rows < 10000:
        val, rule = 100, "under 10k rows — the default of 100 is fine"
    elif rows <= 1_000_000:
        val, rule = max(1, rows // 1000), "10k–1M rows — rows / 1000"
    else:
        val, rule = int(math.sqrt(rows)), "over 1M rows — sqrt(rows)"
    val = max(LIMITS["lists_min"], min(LIMITS["lists_max"], val))
    return {
        "lists": val,
        "rule": rule,
        "probes_start": max(1, int(math.sqrt(val))),
        "two_level": val >= LIMITS["two_level_lists"],
        "rows_per_list": round(rows / val, 1) if val else 0,
        "kmeans_sample": max(50 * val, 10000),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH / STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    try:
        get_conn().close()
        return jsonify({"ok": True, "ts": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/status")
def status():
    out = {
        "db_ok": False, "schema": SCHEMA, "table": TABLE,
        "extension": None, "pgvector": None, "rerank_policy": None,
        "table_exists": False, "rows": 0, "dim": 0,
        "indexes": [], "limits": LIMITS,
        "generator": GEN_STATE, "build": {k: v for k, v in BUILD_STATE.items() if k != "notices"},
    }
    try:
        conn = get_conn()
        conn.close()
        out["db_ok"] = True
    except Exception as e:
        out["error"] = str(e)
        return jsonify(out)

    out["corpus"] = CORPUS_NAME
    out["server_version"] = scalar("SELECT current_setting('server_version')")
    out["extension"] = scalar(
        "SELECT extversion FROM pg_extension WHERE extname = 'edb_vectorplus'")
    out["pgvector"] = scalar(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    out["rerank_policy"] = scalar(
        "SELECT edb_vectorplus_rerank_policy_available()", default=None)
    out["blcksz"] = int(scalar("SELECT current_setting('block_size')", default=8192) or 8192)
    out["maintenance_work_mem"] = scalar("SELECT current_setting('maintenance_work_mem')")
    out["max_parallel_maintenance_workers"] = scalar(
        "SELECT current_setting('max_parallel_maintenance_workers')")

    out["table_exists"] = bool(scalar("SELECT to_regclass(%s) IS NOT NULL",
                                      (f"{SCHEMA}.{TABLE}",), default=False))
    if out["table_exists"]:
        out["dim"] = table_dim()
        out["rows"] = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
        out["table_size"] = scalar("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                   (f"{SCHEMA}.{TABLE}",))
        out["indexes"] = ivfplus_indexes()
        out["packing"] = packing_window(out["dim"], out["blcksz"])
        out["advisor"] = recommend_lists(out["rows"])
        out["scalar_indexes"] = scalar_index_state()
        out["fts_index"] = any(i["key"] == "tsv" and i["present"] for i in out["scalar_indexes"])
    return jsonify(out)


@app.route("/api/env")
def env_info():
    """Everything that decides whether a build succeeds or is quietly slow."""
    keys = ["server_version", "block_size", "maintenance_work_mem",
            "max_parallel_maintenance_workers", "shared_buffers", "work_mem",
            "effective_cache_size", "max_worker_processes"]
    out = {}
    for k in keys:
        out[k] = scalar(f"SELECT current_setting('{k}')")
    out["opclasses"] = OPCLASSES
    out["limits"] = LIMITS
    return jsonify(out)


# ══════════════════════════════════════════════════════════════════════════════
#  SCALAR INDEXES  (tsv GIN + category btree)
# ══════════════════════════════════════════════════════════════════════════════
# These two are built AFTER the bulk load, for the same reason the ivfplus index
# is. GIN maintenance during a COPY is per-row work on a posting-list structure;
# one build pass over the finished table is faster and produces a tighter index.
# The btree matters less, but there is no reason to carry it through the load
# either. The loader drops both up front and rebuilds them when the rows land,
# so a reload never silently runs with the indexes still attached.
#
# Note the tsvector itself is a STORED generated column — to_tsvector() runs on
# every inserted row no matter what. Only the index maintenance moves.

SCALAR_INDEX_SPECS = [
    {
        "key":  "tsv",
        "name": f"{TABLE}_tsv_idx",
        "what": "GIN on tsv — the lexical half of hybrid search",
        "ddl":  f'CREATE INDEX "{TABLE}_tsv_idx" ON {qualified(TABLE)} USING GIN (tsv)',
    },
    {
        "key":  "cat",
        "name": f"{TABLE}_cat_idx",
        "what": "btree on category_id — the metadata filter",
        "ddl":  f'CREATE INDEX "{TABLE}_cat_idx" ON {qualified(TABLE)} (category_id)',
    },
]

# Bumped for the duration of an index build only. A GIN build that spills to
# disk is dramatically slower than one that fits in memory.
INDEX_MAINT_WORK_MEM = os.environ.get("VP_INDEX_MAINT_WORK_MEM", "512MB")
_MWM_RE = re.compile(r"^\d+\s*(kB|MB|GB|B)?$", re.IGNORECASE)


def scalar_index_ddl() -> str:
    """The two CREATE INDEX statements as one block, for display in the UI."""
    return ";\n".join(s["ddl"] for s in SCALAR_INDEX_SPECS) + ";"


def drop_scalar_indexes(cur):
    """Remove both indexes if present. Returns the names actually dropped."""
    dropped = []
    for spec in SCALAR_INDEX_SPECS:
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f'{SCHEMA}.{spec["name"]}',))
        if cur.fetchone()[0]:
            cur.execute(f'DROP INDEX IF EXISTS "{SCHEMA}"."{spec["name"]}"')
            dropped.append(spec["name"])
    return dropped


def build_scalar_indexes(cur):
    """Build both indexes, timing each one. Returns a row per index."""
    if _MWM_RE.match(INDEX_MAINT_WORK_MEM or ""):
        try:
            cur.execute("SET maintenance_work_mem = %s", (INDEX_MAINT_WORK_MEM,))
        except Exception:
            pass  # not fatal — the build just runs with whatever the server has
    out = []
    for spec in SCALAR_INDEX_SPECS:
        cur.execute(f'DROP INDEX IF EXISTS "{SCHEMA}"."{spec["name"]}"')
        t0 = time.perf_counter()
        cur.execute(spec["ddl"])
        seconds = round(time.perf_counter() - t0, 3)
        cur.execute("""SELECT pg_relation_size(oid), pg_size_pretty(pg_relation_size(oid))
                       FROM pg_class WHERE oid = to_regclass(%s)""",
                    (f'{SCHEMA}.{spec["name"]}',))
        row = cur.fetchone() or (0, None)
        out.append({"name": spec["name"], "what": spec["what"], "seconds": seconds,
                    "bytes": int(row[0] or 0), "size": row[1]})
    return out


def scalar_index_state():
    """Presence and size of each scalar index, for the status grid."""
    out = []
    for spec in SCALAR_INDEX_SPECS:
        b = scalar("""SELECT pg_relation_size(oid) FROM pg_class
                      WHERE oid = to_regclass(%s)""",
                   (f'{SCHEMA}.{spec["name"]}',), default=None)
        present = b is not None
        out.append({
            "key": spec["key"], "name": spec["name"], "what": spec["what"],
            "present": present, "bytes": int(b or 0),
            "size": scalar("SELECT pg_size_pretty(%s::bigint)", (int(b or 0),)) if present else None,
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/setup/extension", methods=["POST"])
def setup_extension():
    """CREATE EXTENSION edb_vectorplus CASCADE — CASCADE pulls in pgvector."""
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS edb_vectorplus CASCADE")
        log_query("ddl", "CREATE EXTENSION IF NOT EXISTS edb_vectorplus CASCADE")
        notices = clean_notices(conn.notices)
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='edb_vectorplus'")
        ver = cur.fetchone()
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        pgv = cur.fetchone()
        return jsonify({"ok": True, "version": ver[0] if ver else None,
                        "pgvector": pgv[0] if pgv else None, "notices": notices})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/setup/create_table", methods=["POST"])
def setup_create_table():
    """
    Drop + recreate the demo table at the requested dimension. The tsvector is a
    stored generated column so hybrid search has a real lexical side.

    No indexes are created here. The GIN and btree indexes are built by the
    loader once the rows are in, for the same reason the ivfplus index is: an
    index maintained row-by-row through a COPY costs more than one build pass
    over the finished table.
    """
    body = request.get_json(silent=True) or {}
    dim  = int(body.get("dim", 768))
    if dim < 2 or dim > LIMITS["max_dim_vector"]:
        return jsonify({"ok": False,
                        "error": f"dim must be 2..{LIMITS['max_dim_vector']} for the vector type"}), 400

    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
        cur.execute(f"DROP TABLE IF EXISTS {qualified(TABLE)} CASCADE")
        log_query("ddl", f"DROP TABLE IF EXISTS {qualified(TABLE)} CASCADE")
        cur.execute(f"""
            CREATE TABLE {qualified(TABLE)} (
                id          bigserial PRIMARY KEY,
                topic       text     NOT NULL,
                category_id int      NOT NULL,
                title       text     NOT NULL,
                body        text     NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now(),
                tsv         tsvector GENERATED ALWAYS AS
                            (to_tsvector('english', title || ' ' || body)) STORED,
                embedding   vector({dim})
            )
        """)
        return jsonify({"ok": True, "dim": dim, "packing": packing_window(dim),
                        "indexes_deferred": [s["name"] for s in SCALAR_INDEX_SPECS],
                        "notices": clean_notices(conn.notices)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/setup/scalar_indexes", methods=["POST"])
def setup_scalar_indexes():
    """Build or drop the tsv/category indexes by hand.

    The loader already does this around the COPY, so this is for two cases: a
    load that failed part-way and left the table without them, and showing the
    build cost on its own during a demo.
    """
    body   = request.get_json(silent=True) or {}
    action = (body.get("action") or "build").lower()
    if action not in ("build", "drop"):
        return jsonify({"ok": False, "error": "action must be 'build' or 'drop'"}), 400
    if GEN_STATE["running"]:
        return jsonify({"ok": False,
                        "error": "The loader is running and manages these indexes itself"}), 400
    if not scalar("SELECT to_regclass(%s) IS NOT NULL", (f"{SCHEMA}.{TABLE}",), default=False):
        return jsonify({"ok": False, "error": "Table does not exist — create it first"}), 400

    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        if action == "drop":
            return jsonify({"ok": True, "action": "drop", "dropped": drop_scalar_indexes(cur)})
        rows = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
        detail = build_scalar_indexes(cur)
        cur.execute(f"ANALYZE {qualified(TABLE)}")
        return jsonify({"ok": True, "action": "build", "rows": rows, "detail": detail,
                        "seconds": round(sum(d["seconds"] for d in detail), 2),
                        "maintenance_work_mem": INDEX_MAINT_WORK_MEM})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/setup/reset", methods=["POST"])
def setup_reset():
    """Truncate rows, keep the table and its indexes. The ivfplus index surviving
    an empty table is the point of the health tab: an index whose rows all went
    away is not a fresh index. The scalar indexes survive too, but the loader
    drops them before the next COPY, so a reload still loads unindexed."""
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute(f"TRUNCATE {qualified(TABLE)} RESTART IDENTITY")
        log_query("ddl", f"TRUNCATE {qualified(TABLE)} RESTART IDENTITY")
        BUILD_HISTORY.clear()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS GENERATION  (build vectors from the UI)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_worker(total: int, dim: int, batch: int, seed: int, use_binary: bool = True):
    GEN_STATE.update({"running": True, "done": 0, "total": total, "error": None,
                      "started": time.time(), "phase": "preparing", "dim": dim,
                      "rows_per_sec": 0, "elapsed": 0,
                      "copy_seconds": None, "index_seconds": None,
                      "dropped_indexes": [], "index_detail": [],
                      "copy_format": None})
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()

        # Indexes come off before the load, not after the fact. On a reload this
        # is what stops the second run from quietly paying GIN maintenance per
        # row and reporting a worse number than the first.
        GEN_STATE["dropped_indexes"] = drop_scalar_indexes(cur)

        GEN_STATE["phase"] = "generating"
        rng = np.random.default_rng(seed)
        pyrng = random.Random(seed)
        remaining = total
        t_copy = time.perf_counter()
        while remaining > 0 and GEN_STATE["running"]:
            n = min(batch, remaining)
            rows = make_batch(rng, dim, n, pyrng=pyrng)
            fmt = copy_rows(cur, rows, binary=use_binary)
            if use_binary and fmt == "text":
                use_binary = False      # asked once, refused — stop retrying
            GEN_STATE["copy_format"] = fmt
            remaining -= n
            GEN_STATE["done"] += n
            el = max(0.001, time.time() - GEN_STATE["started"])
            GEN_STATE["elapsed"] = round(el, 2)
            GEN_STATE["rows_per_sec"] = int(GEN_STATE["done"] / el)
        GEN_STATE["copy_seconds"] = round(time.perf_counter() - t_copy, 2)

        # Build the indexes even if the operator hit Stop — leaving the table
        # without them would break hybrid search and the filter tab.
        GEN_STATE["phase"] = "indexing"
        detail = build_scalar_indexes(cur)
        GEN_STATE["index_detail"] = detail
        GEN_STATE["index_seconds"] = round(sum(d["seconds"] for d in detail), 2)

        GEN_STATE["phase"] = "analyzing"
        cur.execute(f"ANALYZE {qualified(TABLE)}")
        GEN_STATE["phase"] = "done"
    except Exception as e:
        GEN_STATE["error"] = str(e)
        GEN_STATE["phase"] = "error"
    finally:
        GEN_STATE["running"] = False
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/data/generate/start", methods=["POST"])
def data_generate_start():
    if GEN_STATE["running"]:
        return jsonify({"ok": False, "error": "Generation already running"}), 400
    if BUILD_STATE["running"]:
        return jsonify({"ok": False, "error": "An index build is running — wait for it"}), 400

    body  = request.get_json(silent=True) or {}
    total = max(1, min(5_000_000, int(body.get("rows", 50000))))
    batch = max(500, min(20000, int(body.get("batch", 5000))))
    seed  = int(body.get("seed", 42))

    dim = table_dim()
    if not dim:
        return jsonify({"ok": False, "error": "Table does not exist — create it first"}), 400

    warn = []
    idxs = ivfplus_indexes()
    if idxs:
        warn.append("An ivfplus index already exists. Clustering happens at build "
                    "time only, so these rows land in the append region and are read "
                    "by the slower bit-sliced scan. REINDEX afterwards, or drop the "
                    "index, load, and rebuild.")

    use_binary = bool(body.get("binary", COPY_BINARY_DEFAULT))

    t = threading.Thread(target=_generate_worker,
                         args=(total, dim, batch, seed, use_binary), daemon=True)
    t.start()
    return jsonify({"ok": True, "total": total, "dim": dim, "warnings": warn,
                    "binary": use_binary})


@app.route("/api/data/generate/status")
def data_generate_status():
    s = dict(GEN_STATE)
    if s["running"] and s["started"]:
        s["elapsed"] = round(time.time() - s["started"], 2)
        if s["total"] and s["done"]:
            rate = s["done"] / max(0.001, s["elapsed"])
            s["eta"] = round((s["total"] - s["done"]) / max(1, rate), 1)
    s["pct"] = round(100 * s["done"] / s["total"], 1) if s["total"] else 0
    return jsonify(s)


@app.route("/api/data/generate/stop", methods=["POST"])
def data_generate_stop():
    GEN_STATE["running"] = False
    return jsonify({"ok": True})


@app.route("/api/data/sample")
def data_sample():
    n = max(1, min(50, int(request.args.get("n", 8))))
    try:
        rows = q(f"""SELECT id, topic, category_id, title, left(body, 90) AS body,
                            substring(embedding::text, 1, 46) || '…' AS emb_head
                     FROM {qualified(TABLE)} ORDER BY id DESC LIMIT %s""", (n,))
        return jsonify({"ok": True, "rows": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/data/topics")
def data_topics():
    try:
        rows = q(f"""SELECT topic, category_id, count(*) AS n
                     FROM {qualified(TABLE)} GROUP BY topic, category_id ORDER BY n DESC""")
    except Exception:
        rows = []
    return jsonify({"ok": True, "rows": rows,
                    "catalog": [{"topic": k, "category_id": v["category_id"],
                                 "words": v["words"][:10]} for k, v in TOPICS.items()]})


# ══════════════════════════════════════════════════════════════════════════════
#  INDEX BUILD  (with live progress + wall-clock timing)
# ══════════════════════════════════════════════════════════════════════════════

def _progress_watcher(pid: int):
    """pg_stat_progress_create_index, polled from a second connection."""
    while BUILD_STATE["running"]:
        try:
            r = q1("""SELECT phase, blocks_done, blocks_total,
                             tuples_done, tuples_total, partitions_done, partitions_total
                      FROM pg_stat_progress_create_index WHERE pid = %s""",
                   (pid,), timeout_ms=3000)
            if r:
                BUILD_STATE["progress"] = r
        except Exception:
            pass
        time.sleep(0.5)


def _build_worker(sql: str, name: str, lists: int, opclass: str, concurrently: bool,
                  maint_mem: str = None):
    BUILD_STATE.update({"running": True, "sql": sql, "started": time.time(),
                        "elapsed": 0, "error": None, "notices": [],
                        "phase": "building", "progress": None, "result": None,
                        "maintenance_work_mem": maint_mem})
    conn = None
    rows_before = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    est = ivf_build_memory(rows_before, table_dim(), lists)
    est_mb = est["mb"]
    BUILD_STATE["retried"] = None
    try:
        conn = get_conn(autocommit=True)          # CONCURRENTLY needs autocommit
        cur = conn.cursor()
        cur.execute("SELECT pg_backend_pid()")
        pid = cur.fetchone()[0]
        BUILD_STATE["pid"] = pid
        threading.Thread(target=_progress_watcher, args=(pid,), daemon=True).start()

        cur.execute("SET statement_timeout = 0")
        # The k-means step allocates its whole sample up front. Without this the
        # build fails immediately against the server default, which is usually
        # 64MB — far below what any realistic lists value needs.
        applied_mem = maint_mem
        required_mb = None
        cap_b = parse_mem_setting(BUILD_MAX_MAINT_MEM) or (4 * 1024 ** 3)

        for attempt in (1, 2):
            if applied_mem and _MWM_RE.match(applied_mem):
                cur.execute("SET maintenance_work_mem = %s", (applied_mem,))
            t0 = time.perf_counter()
            try:
                cur.execute(sql)
                elapsed = time.perf_counter() - t0
                log_query("index-build", sql,
                          settings=[f"maintenance_work_mem = {applied_mem}"] if applied_mem else None,
                          ms=elapsed * 1000, rows=rows_before, label=name)
                break
            except Exception as e:
                hit = MEM_ERROR_RE.search(str(e))
                # The server just told us the real number. Believe it over the
                # model, and retry once rather than making the user do it.
                if hit and attempt == 1:
                    required_mb = int(hit.group(1))
                    need = int(math.ceil(required_mb * 1.1 * 1024 * 1024
                                         / 67_108_864)) * 67_108_864
                    if need > cap_b:
                        raise RuntimeError(
                            f"The server requires {required_mb} MB for this build, above the "
                            f"{BUILD_MAX_MAINT_MEM} ceiling this app will set "
                            f"(VP_BUILD_MAX_MAINT_MEM). Lower lists to "
                            f"{max_lists_for_memory(cap_b, table_dim(), rows_before) :,}, "
                            f"or raise the ceiling.")
                    applied_mem = fmt_mb(need)
                    BUILD_STATE["retried"] = {
                        "estimated_mb": est_mb, "required_mb": required_mb,
                        "retrying_at": applied_mem}
                    BUILD_STATE["maintenance_work_mem"] = applied_mem
                    continue
                raise
        maint_mem = applied_mem

        BUILD_STATE["notices"] = clean_notices(conn.notices)
        BUILD_STATE["elapsed"] = round(elapsed, 3)
        BUILD_STATE["phase"] = "done"

        size_b = scalar("SELECT pg_relation_size(%s::regclass)",
                        (f"{SCHEMA}.{name}",), default=0) or 0
        rows   = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
        health = index_health(name)

        rec = {
            "name": name, "lists": lists, "opclass": opclass,
            "concurrently": concurrently,
            "seconds": round(elapsed, 3),
            "rows": rows,
            "rows_per_sec": int(rows / elapsed) if elapsed > 0 else 0,
            "bytes": int(size_b),
            "size": scalar("SELECT pg_size_pretty(%s::bigint)", (int(size_b),)),
            "bytes_per_row": round(size_b / rows, 1) if rows else 0,
            "two_level": lists >= LIMITS["two_level_lists"],
            "maintenance_work_mem": maint_mem,
            "mem_estimated_mb": est_mb,
            "mem_required_mb": required_mb,
            "notices": BUILD_STATE["notices"],
            "health": health.get("index_health"),
            "at": datetime.now().strftime("%H:%M:%S"),
            "sql": sql,
        }
        rec["target"] = TARGET_NAME
        rec["id"] = store_insert(
            "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target=TARGET_NAME, label=TARGET_LABEL, engine="ivfplus", am="ivfplus",
            dsn_db=DB_CONFIG.get("database"), rows=rows, dim=table_dim(),
            lists=lists, opclass=opclass, rotation=1,
            concurrently=int(bool(concurrently)), index_name=name,
            build_seconds=round(elapsed, 3),
            rows_per_sec=round(rows / elapsed, 1) if elapsed > 0 else 0,
            index_bytes=int(size_b),
            bytes_per_row=round(size_b / rows, 2) if rows else 0,
            maint_work_mem=maint_mem, mem_estimated_mb=est_mb,
            mem_required_mb=required_mb,
            parallel_workers=scalar("SELECT current_setting('max_parallel_maintenance_workers')"),
            health=rec.get("health"),
            pg_version=scalar("SELECT current_setting('server_version')"),
            ext_version=scalar("SELECT extversion FROM pg_extension "
                               "WHERE extname = 'edb_vectorplus'"),
            notices=" | ".join(BUILD_STATE["notices"]), note=None)
        BUILD_STATE["result"] = rec
        BUILD_HISTORY.append(rec)
    except Exception as e:
        msg = str(e)
        if "maintenance_work_mem" in msg and "memory required" in msg:
            msg += ("\n\nThe k-means step allocates its whole sample before it starts. "
                    "Raise maintenance_work_mem for the build, or lower lists — the "
                    "build panel shows the largest lists value that fits.")
        BUILD_STATE["error"] = msg
        BUILD_STATE["phase"] = "error"
        if conn:
            BUILD_STATE["notices"] = clean_notices(conn.notices)
    finally:
        BUILD_STATE["running"] = False
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/index/build/start", methods=["POST"])
def index_build_start():
    if BUILD_STATE["running"]:
        return jsonify({"ok": False, "error": "A build is already running"}), 400
    if GEN_STATE["running"]:
        return jsonify({"ok": False, "error": "Data generation is running — "
                                              "load finishes before the build starts"}), 400

    body         = request.get_json(silent=True) or {}
    lists        = int(body.get("lists", 100))
    opclass      = body.get("opclass", "vector_l2_ops")
    rotation     = bool(body.get("rotation", True))
    store_vec    = body.get("store_vectors", None)
    concurrently = bool(body.get("concurrently", False))
    am           = body.get("am", "ivfplus")          # ivfplus | ivfflat, for comparison
    where        = (body.get("where") or "").strip()

    name = (body.get("name") or "").strip()
    if name:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,60}", name):
            return jsonify({"ok": False,
                            "error": "Index name must be letters, digits and underscores"}), 400
    else:
        # Comparing builds means building several — auto-uniquify so a second
        # build at the same lists does not collide with the first.
        short = opclass.replace("vector_", "v").replace("halfvec_", "h").replace("_ops", "")
        base  = f"{IDX_PREFIX}_{am}_l{lists}_{short}"
        taken = {r["name"] for r in ivfplus_indexes()}
        name  = base
        n = 2
        while name in taken:
            name = f"{base}_{n}"
            n += 1

    if opclass not in OPCLASSES:
        return jsonify({"ok": False, "error": f"Unknown opclass {opclass}"}), 400
    if not (LIMITS["lists_min"] <= lists <= LIMITS["lists_max"]):
        return jsonify({"ok": False,
                        "error": f"lists must be {LIMITS['lists_min']}–{LIMITS['lists_max']}"}), 400

    rows = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    if rows == 0:
        return jsonify({"ok": False, "error": "Table is empty. Load data BEFORE "
                                              "CREATE INDEX — the build clusters the "
                                              "rows that are already there."}), 400

    if am == "ivfflat" and opclass.startswith("halfvec"):
        pass  # pgvector supports halfvec opclasses too; leave it to the server

    col = "embedding"
    if opclass.startswith("halfvec"):
        col = f"(embedding::halfvec({table_dim()}))"

    opts = [f"lists = {lists}"]
    if am == "ivfplus":
        opts.append(f"rotation = {'on' if rotation else 'off'}")
        if store_vec is not None:
            opts.append(f"store_vectors = {'on' if store_vec else 'off'}")

    sql = (f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}"
           f'"{name}" ON {qualified(TABLE)}\n'
           f"    USING {am} ({col} {opclass})\n"
           f"    WITH ({', '.join(opts)})")
    if where:
        sql += f"\n    WHERE {where}"

    warnings = []

    # ── Build-time memory ────────────────────────────────────────────────────
    # k-means holds max(50 x lists, 10000) sample rows in memory at once. Against
    # the usual 64MB default this fails on the spot for any realistic lists, so
    # size the session setting here rather than letting the build die.
    mem        = ivf_build_memory(rows, table_dim(), lists)
    server_mwm = scalar("SELECT current_setting('maintenance_work_mem')")
    server_b   = parse_mem_setting(server_mwm) or 0
    asked      = (body.get("maintenance_work_mem") or "").strip()
    cap_b      = parse_mem_setting(BUILD_MAX_MAINT_MEM) or (2 * 1024 ** 3)

    if asked:
        if not _MWM_RE.match(asked):
            return jsonify({"ok": False,
                            "error": "maintenance_work_mem must look like 512MB or 2GB"}), 400
        maint_mem = asked
    elif mem["bytes"] > server_b:
        # Round up to the next 64MB so the number is a tidy one to repeat.
        need = min(cap_b, ((mem["bytes"] + 67_108_863) // 67_108_864) * 67_108_864)
        maint_mem = fmt_mb(need)
    else:
        maint_mem = None                       # server default is already enough

    effective_b = parse_mem_setting(maint_mem) if maint_mem else server_b
    if mem["bytes"] > effective_b:
        fits = max_lists_for_memory(effective_b, table_dim(), rows)
        return jsonify({
            "ok": False,
            "error": (f"This build needs about {mem['pretty']} of memory — the k-means step "
                      f"samples {mem['samples']:,} rows at {table_dim()} dimensions — but "
                      f"maintenance_work_mem is {maint_mem or server_mwm}"
                      + (f", and the app will not raise it past {BUILD_MAX_MAINT_MEM}"
                         if not asked else "") + ".\n\n"
                      f"Either set maintenance_work_mem to {mem['pretty']} or more for this "
                      f"build, or drop lists to {fits:,}, which fits in "
                      f"{fmt_mb(effective_b)}."),
            "memory": mem, "max_lists": fits,
            "server_maintenance_work_mem": server_mwm,
        }), 400

    if maint_mem and mem["bytes"] > server_b:
        warnings.append(f"This build needs about {mem['pretty']} (k-means samples "
                        f"{mem['samples']:,} rows), more than the server's "
                        f"{server_mwm}. maintenance_work_mem is set to {maint_mem} "
                        f"for this session only — it does not change the server config.")
    elif maint_mem:
        warnings.append(f"maintenance_work_mem is set to {maint_mem} for this session "
                        f"only. The build needs about {mem['pretty']}.")
    if lists >= LIMITS["two_level_lists"]:
        warnings.append(f"lists >= {LIMITS['two_level_lists']} switches to the two-level "
                        f"hierarchy, and that build is single-threaded — you forfeit "
                        f"max_parallel_maintenance_workers. Budget for a longer build.")
    sample = max(50 * lists, 10000)
    if sample > rows:
        warnings.append(f"k-means samples max(50 × lists, 10000) = {sample:,} rows but the "
                        f"table has {rows:,}. Expect the 'created with little data' NOTICE "
                        f"and centroids that will not generalize.")
    if rows / max(1, lists) > 5000:
        warnings.append(f"{rows/max(1,lists):,.0f} rows per list. Probing one list means "
                        f"reading a lot — raise lists.")
    if not concurrently:
        warnings.append("CREATE INDEX blocks writes on the table. Production builds should "
                        "use CREATE INDEX CONCURRENTLY.")
    pk = packing_window(table_dim())
    if not pk["packed"] and am == "ivfplus":
        warnings.append("Outside the packing window: this index runs the slower unpacked "
                        "path and health will report 1.0 forever. " + "; ".join(pk["reasons"]))

    t = threading.Thread(target=_build_worker,
                         args=(sql, name, lists, opclass, concurrently, maint_mem),
                         daemon=True)
    t.start()
    return jsonify({"ok": True, "sql": sql, "warnings": warnings,
                    "advisor": recommend_lists(rows),
                    "memory": mem, "maintenance_work_mem": maint_mem})


@app.route("/api/index/build/status")
def index_build_status():
    s = dict(BUILD_STATE)
    if s["running"] and s["started"]:
        s["elapsed"] = round(time.time() - s["started"], 2)
    return jsonify(s)


@app.route("/api/index/list")
def index_list():
    out = []
    for r in ivfplus_indexes():
        if r["am"] == "ivfplus":
            r["health"] = index_health(r["name"])
        out.append(r)
    hist = store_query("SELECT * FROM builds WHERE target = ? ORDER BY id DESC LIMIT 50",
                       (TARGET_NAME,))
    # Shape the persisted rows like the in-session ones the build panel renders.
    for h in hist:
        h["at"] = (h.get("ts") or "")[5:16]
        h["seconds"] = h.get("build_seconds")
        h["bytes"] = h.get("index_bytes")
        h["size"] = human_bytes(h.get("index_bytes"))
    return jsonify({"ok": True, "rows": out, "history": list(reversed(hist))})


@app.route("/api/index/drop", methods=["POST"])
def index_drop():
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    conc = bool(body.get("concurrently", False))
    if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        return jsonify({"ok": False, "error": "Bad index name"}), 400
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        _ds = f'DROP INDEX {"CONCURRENTLY " if conc else ""}IF EXISTS "{SCHEMA}"."{name}"'
        cur.execute(_ds)
        log_query("ddl", _ds, label=name)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/index/advisor")
def index_advisor():
    rows = int(request.args.get("rows") or
               scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    dim = table_dim()
    lists = int(request.args.get("lists") or 0)
    rec = recommend_lists(rows)
    rec["rows"] = rows
    rec["packing"] = packing_window(dim)
    server_mwm = scalar("SELECT current_setting('maintenance_work_mem')")
    server_b   = parse_mem_setting(server_mwm) or 0
    rec["maintenance_work_mem"] = server_mwm
    rec["build_memory"] = ivf_build_memory(rows, dim, lists or rec["lists"])
    rec["lists_fitting_server_mwm"] = max_lists_for_memory(server_b, dim, rows)
    rec["build_mem_cap"] = BUILD_MAX_MAINT_MEM
    return jsonify(rec)


@app.route("/api/index/size_model")
def index_size_model():
    """
    The doc's per-row size table, evaluated for the live table. The f16 rerank
    copy is ~92% of a stock build; a rerank-policy server drops it by default.
    """
    dim   = table_dim()
    rows  = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    lists = int(request.args.get("lists", 100))
    if not dim:
        return jsonify({"ok": False, "error": "No table"}), 400

    packed_row = 1028 / 32 + dim / 8
    f16_row    = 6 + 2 * dim + 4
    centroids  = lists * dim * 4
    stock      = rows * (packed_row + f16_row) + centroids
    policy     = rows * packed_row + centroids
    return jsonify({
        "ok": True, "dim": dim, "rows": rows, "lists": lists,
        "packed_bytes_per_row": round(packed_row, 1),
        "f16_bytes_per_row": round(f16_row, 1),
        "centroid_bytes": centroids,
        "stock_total": int(stock), "policy_total": int(policy),
        "f16_share_pct": round(100 * rows * f16_row / stock, 1) if stock else 0,
        "append_bytes_per_row": 40 + 8 * math.ceil(dim / 64) + 4,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def resolve_query_vector(body):
    """Text typed in the UI, an explicit vector, or a row already in the table."""
    dim = table_dim()
    if body.get("vector"):
        v = np.array(body["vector"], dtype=np.float32)
        return v, "raw vector"
    if body.get("id"):
        r = q1(f"SELECT embedding::text AS e, title FROM {qualified(TABLE)} WHERE id = %s",
               (int(body["id"]),))
        if not r:
            raise ValueError(f"No row with id {body['id']}")
        v = np.array([float(x) for x in r["e"].strip("[]").split(",")], dtype=np.float32)
        return v, f"row #{body['id']} — {r['title']}"
    text = (body.get("text") or "").strip()
    if not text:
        raise ValueError("Provide text, a vector, or a row id")
    return embed_text(text, dim), f"text: {text}"


def run_search(qv, opclass, probes, limit, where=None, iterative=None,
               max_probes=None, explain=False, exact=False, log_kind=None):
    op   = OPCLASSES[opclass]["op"]
    col  = "embedding"
    cast = ""
    if opclass.startswith("halfvec"):
        col  = f"embedding::halfvec({table_dim()})"
        cast = f"::halfvec({table_dim()})"

    lit = vec_literal(qv)
    sql = (f"SELECT id, topic, category_id, title,\n"
           f"       {col} {op} %s{cast} AS distance\n"
           f"FROM {qualified(TABLE)}\n")
    params = [lit]
    if where:
        sql += f"WHERE {where}\n"
    sql += f"ORDER BY {col} {op} %s{cast}\nLIMIT {int(limit)}"
    params.append(lit)

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SET LOCAL statement_timeout = 60000")
        settings = []
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
            settings.append("enable_indexscan = off")
        else:
            cur.execute("SET LOCAL ivfplus.probes = %s", (int(probes),))
            settings.append(f"ivfplus.probes = {probes}")
            if iterative and iterative != "off":
                # Never enable this without lowering max_probes in the same breath:
                # here it only widens the INITIAL probe set, it never refills.
                mp = int(max_probes or 128)
                cur.execute("SET LOCAL ivfplus.max_probes = %s", (mp,))
                cur.execute("SET LOCAL ivfplus.iterative_scan = %s", (iterative,))
                settings.append(f"ivfplus.max_probes = {mp}")
                settings.append(f"ivfplus.iterative_scan = {iterative}")

        plan = None
        if explain:
            cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql, params)
            plan = "\n".join(r["QUERY PLAN"] for r in cur.fetchall())

        t0 = time.perf_counter()
        try:
            cur.execute(sql, params)
        except Exception as e:
            log_query(log_kind or ("knn-exact" if exact else "knn"), sql,
                      settings=settings, error=e,
                      sql_full=resolve_sql(cur, sql, params))
            raise
        rows = [dict(r) for r in cur.fetchall()]
        ms = (time.perf_counter() - t0) * 1000.0
        conn.commit()
        if log_kind != "__skip__":
            log_query(log_kind or ("knn-exact" if exact else "knn"), sql,
                      settings=settings, ms=ms, rows=len(rows),
                      sql_full=resolve_sql(cur, sql, params))
        for r in rows:
            r["distance"] = round(float(r["distance"]), 6)
        return {"rows": rows, "ms": round(ms, 2), "sql": sql,
                "settings": settings, "plan": plan}
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/search/knn", methods=["POST"])
def search_knn():
    body = request.get_json(silent=True) or {}
    try:
        qv, label = resolve_query_vector(body)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    opclass    = body.get("opclass", "vector_l2_ops")
    probes     = int(body.get("probes", 10))
    limit      = int(body.get("limit", 10))
    where      = (body.get("where") or "").strip() or None
    iterative  = body.get("iterative_scan", "off")
    max_probes = int(body.get("max_probes", 128))
    explain    = bool(body.get("explain", False))

    if opclass not in OPCLASSES:
        return jsonify({"ok": False, "error": "Unknown opclass"}), 400

    notes = []
    if limit > LIMITS["rerank_pool"]:
        notes.append(f"LIMIT {limit} is above the fixed {LIMITS['rerank_pool']}-row rerank "
                     f"pool. A vector-opclass scan returns at most "
                     f"{LIMITS['rerank_pool']} rows per query, so you will get fewer "
                     f"rows than you asked for. There is no runtime knob for this.")
    if where:
        notes.append("The filter is applied AFTER the index scan, as with pgvector's "
                     "IVFFlat — a selective filter can return fewer than LIMIT rows. "
                     "Use a partial index or widen probes.")
    if iterative and iterative != "off":
        notes.append(f"iterative_scan here only widens the INITIAL probe set to "
                     f"max_probes ({max_probes}); it never refills. Left at its default "
                     f"of 32768 it clamps to every list in the index — a full up-front scan.")

    try:
        res = run_search(qv, opclass, probes, limit, where, iterative, max_probes, explain)
        res.update({"ok": True, "label": label, "notes": notes,
                    "returned": len(res["rows"]), "requested": limit,
                    "operator": OPCLASSES[opclass]["op"],
                    "metric": OPCLASSES[opclass]["metric"]})
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/search/compare", methods=["POST"])
def search_compare():
    """Index scan vs a forced exact scan, side by side, with the overlap."""
    body = request.get_json(silent=True) or {}
    try:
        qv, label = resolve_query_vector(body)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    opclass = body.get("opclass", "vector_l2_ops")
    probes  = int(body.get("probes", 10))
    limit   = int(body.get("limit", 10))
    try:
        exact = run_search(qv, opclass, probes, limit, exact=True, log_kind="compare-exact")
        appr  = run_search(qv, opclass, probes, limit, exact=False, log_kind="compare-index")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    tids = {r["id"] for r in exact["rows"]}
    gids = {r["id"] for r in appr["rows"]}
    hit  = tids & gids
    return jsonify({
        "ok": True, "label": label, "probes": probes, "k": limit,
        "exact": exact, "approx": appr,
        "recall": round(len(hit) / len(tids), 4) if tids else 0,
        "speedup": round(exact["ms"] / appr["ms"], 1) if appr["ms"] > 0 else None,
        "hits": sorted(hit),
    })


@app.route("/api/search/capabilities")
def search_capabilities():
    """The supported query surface, as runnable snippets for the UI."""
    d = table_dim() or 768
    caps = [
        {"id": "knn_l2", "title": "K-NN, L2", "opclass": "vector_l2_ops",
         "sql": f"SET ivfplus.probes = 32;\nSELECT id, title\nFROM {qualified(TABLE)}\n"
                f"ORDER BY embedding <-> '[…]'\nLIMIT 10;",
         "note": "The default opclass. A bare USING ivfplus (col) works for vector."},
        {"id": "knn_cosine", "title": "K-NN, cosine", "opclass": "vector_cosine_ops",
         "sql": f"SELECT id, title, embedding <=> '[…]' AS dist\nFROM {qualified(TABLE)}\n"
                f"ORDER BY embedding <=> '[…]'\nLIMIT 10;",
         "note": "Needs an index built with vector_cosine_ops. Zero vectors are never "
                 "indexed under cosine — same as pgvector."},
        {"id": "knn_ip", "title": "K-NN, inner product", "opclass": "vector_ip_ops",
         "sql": f"SELECT id, title\nFROM {qualified(TABLE)}\n"
                f"ORDER BY embedding <#> '[…]'\nLIMIT 10;",
         "note": "<#> is negative inner product, so ascending order is still 'closest first'."},
        {"id": "halfvec", "title": "halfvec", "opclass": "halfvec_l2_ops",
         "sql": f"CREATE INDEX … USING ivfplus ((embedding::halfvec({d})) halfvec_l2_ops)\n"
                f"    WITH (lists = 1000);",
         "note": "Indexed exactly like vector — same codes, same rerank copy. Centroids "
                 "are f16, which is why halfvec reaches 4000 dims where vector stops at 2000. "
                 "No default opclass, so the opclass must be named."},
        {"id": "filter", "title": "Filtered", "opclass": "vector_l2_ops",
         "sql": f"SELECT * FROM {qualified(TABLE)}\nWHERE category_id = 2\n"
                f"ORDER BY embedding <-> '[…]'\nLIMIT 5;",
         "note": "Post-scan filter — can return fewer than LIMIT rows."},
        {"id": "partial", "title": "Partial index", "opclass": "vector_l2_ops",
         "sql": f"CREATE INDEX … ON {qualified(TABLE)}\n"
                f"    USING ivfplus (embedding vector_l2_ops)\n    WHERE (category_id = 2);",
         "note": "The fix for a selective, known-ahead filter."},
        {"id": "unsupported", "title": "Not indexed", "opclass": None,
         "sql": "-- <+> (L1), <%> (Jaccard) and sparsevec fall back to a sequential scan.\n"
                "-- bit vectors: use pgvector's own AM instead —\n"
                "--   CREATE INDEX … USING ivfflat (col bit_hamming_ops);",
         "note": "ivfplus deliberately dropped the bit opclass: there is no residual to "
                 "quantize on a bit vector, so it would have been the same exact scan."},
    ]
    return jsonify({"ok": True, "caps": caps, "opclasses": OPCLASSES})


# ══════════════════════════════════════════════════════════════════════════════
#  RECALL LAB
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/recall/sweep", methods=["POST"])
def recall_sweep():
    """
    The doc's recall recipe, swept across probes values: exact top-k with
    enable_indexscan off, then index top-k at each setting, averaged over
    n sampled query vectors.
    """
    body     = request.get_json(silent=True) or {}
    opclass  = body.get("opclass", "vector_l2_ops")
    k        = max(1, min(LIMITS["rerank_pool"], int(body.get("k", 10))))
    nq       = max(1, min(200, int(body.get("queries", 20))))
    plist    = body.get("probes_list") or [1, 2, 4, 8, 16, 32, 64, 128]
    plist    = sorted({max(1, min(LIMITS["probes_max"], int(p))) for p in plist})

    if opclass not in OPCLASSES:
        return jsonify({"ok": False, "error": "Unknown opclass"}), 400
    rows = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    if rows < k:
        return jsonify({"ok": False, "error": "Not enough rows"}), 400

    # Sample query vectors by drawing random ids out of the live id range —
    # far cheaper than ORDER BY random() once the table is large.
    bounds = q1(f"SELECT min(id) AS lo, max(id) AS hi FROM {qualified(TABLE)}")
    lo, hi = int(bounds["lo"]), int(bounds["hi"])
    rng = np.random.default_rng(7)
    ids = sorted({int(x) for x in rng.integers(lo, hi + 1, size=nq * 3)})[: nq * 3]
    samples = q(f"SELECT embedding::text AS e FROM {qualified(TABLE)} "
                f"WHERE id = ANY(%s) LIMIT %s", (ids, nq))
    if not samples:
        return jsonify({"ok": False, "error": "Could not sample query vectors"}), 400

    qvs = []
    for s in samples:
        v = np.array([float(x) for x in s["e"].strip("[]").split(",")], dtype=np.float32)
        # Perturb so the query is near — but not equal to — a stored row.
        v = v + rng.standard_normal(v.shape[0]).astype(np.float32) * 0.05
        n = np.linalg.norm(v)
        qvs.append(v / n if n > 0 else v)
    nq = len(qvs)

    op = OPCLASSES[opclass]["op"]
    col = "embedding"
    cast = ""
    if opclass.startswith("halfvec"):
        col = f"embedding::halfvec({table_dim()})"
        cast = f"::halfvec({table_dim()})"
    qsql = (f"SELECT id FROM {qualified(TABLE)} "
            f"ORDER BY {col} {op} %s{cast} LIMIT {k}")

    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")

        # One resolved example, so the sweep entry is still re-runnable even
        # though the individual statements are not logged one by one.
        _sweep_example = None

        def topk(v, probes=None, exact=False):
            nonlocal _sweep_example
            if exact:
                cur.execute("SET enable_indexscan = off")
                cur.execute("SET enable_bitmapscan = off")
            else:
                cur.execute("SET enable_indexscan = on")
                cur.execute("SET enable_bitmapscan = on")
                cur.execute("SET ivfplus.probes = %s", (int(probes),))
            t0 = time.perf_counter()
            cur.execute(qsql, (vec_literal(v),))
            got = {r[0] for r in cur.fetchall()}
            if _sweep_example is None:
                _sweep_example = resolve_sql(cur, qsql, (vec_literal(v),))
            return got, (time.perf_counter() - t0) * 1000.0

        # Ground truth once per query.
        truth, exact_ms = [], []
        for v in qvs:
            got, ms = topk(v, exact=True)
            truth.append(got)
            exact_ms.append(ms)

        results = []
        for p in plist:
            rec_sum, ms_sum, ret_sum = 0.0, 0.0, 0
            for v, t in zip(qvs, truth):
                got, ms = topk(v, probes=p)
                rec_sum += len(t & got) / max(1, len(t))
                ms_sum += ms
                ret_sum += len(got)
            results.append({
                "probes": p,
                "recall": round(rec_sum / nq, 4),
                "ms": round(ms_sum / nq, 2),
                "avg_returned": round(ret_sum / nq, 1),
            })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass

    exact_avg = round(sum(exact_ms) / len(exact_ms), 2)
    for r in results:
        r["speedup"] = round(exact_avg / r["ms"], 1) if r["ms"] > 0 else None

    # A sweep runs nq x len(probes) statements. Logging each would bury every
    # other entry, so record the shape of the sweep and the query form once.
    log_query("recall-sweep", qsql, sql_full=_sweep_example,
              settings=[f"probes swept over {[r['probes'] for r in results]}",
                        f"k = {k}", f"{nq} query vectors"],
              ms=exact_avg, rows=len(results) * nq,
              label=f"{opclass} · exact baseline {exact_avg}ms")

    # Persist so the curve can be compared against other engines and other days.
    last = store_query("SELECT id, index_name FROM builds WHERE target = ? "
                       "ORDER BY id DESC LIMIT 1", (TARGET_NAME,))
    bid = last[0]["id"] if last else None
    iname = last[0]["index_name"] if last else None
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in results:
        store_insert("recalls", ts=stamp, build_id=bid, target=TARGET_NAME,
                     label=TARGET_LABEL, engine="ivfplus", index_name=iname,
                     rows=rows, dim=table_dim(), k=k, queries=nq,
                     param="ivfplus.probes", value=r["probes"], recall=r["recall"],
                     ms=r["ms"], avg_returned=r["avg_returned"],
                     exact_ms=exact_avg, speedup=r["speedup"], note=None)

    note = ("Averaged over %d sampled query vectors — the reference recommends at "
            "least 50. Re-measure after any lists change." % nq)
    return jsonify({"ok": True, "k": k, "queries": nq, "opclass": opclass,
                    "exact_ms": exact_avg, "results": results, "note": note,
                    "rows": rows})


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID SEARCH  (vector + full text, fused with RRF)
# ══════════════════════════════════════════════════════════════════════════════
# Replaces everything in vectorplus_api.py from the HYBRID_SQL_TEMPLATE
# assignment down to the end of hybrid_search(). Four changes:
#
#   1. Candidate depth defaults to the rerank-pool cap, not 50. This is the
#      setting that decides whether fusion can find agreement at all.
#   2. The lexical leg breaks ties on id, so ranks are repeatable between runs.
#   3. The displayed legs and the fused legs are now the SAME queries at the
#      SAME depth, so the ranks on screen match the ranks in the score.
#   4. Overlap is measured across the whole candidate set, not the top k, so
#      the counter agrees with what fusion actually did.

HYBRID_SQL_TEMPLATE = """\
-- Reciprocal Rank Fusion: composed at the SQL level exactly as with pgvector.
-- The access method does not enter into it.
--
-- Candidate depth ({n}) decides whether fusion has anything to do. The legs
-- only ever agree BELOW their own top-{k}: with a short candidate list RRF
-- degenerates into interleaving two lists it cannot compare.
WITH semantic AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding {op} :qv) AS rank
    FROM {tbl}
    ORDER BY embedding {op} :qv     -- no id tie-break here on purpose: a second
    LIMIT {n}                       -- sort key costs the index-ordered scan
),
lexical AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, query) DESC, id) AS rank
    FROM {tbl}, plainto_tsquery('english', :q) query
    WHERE tsv @@ query
    ORDER BY ts_rank_cd(tsv, query) DESC, id   -- ts_rank_cd ties constantly;
    LIMIT {n}                                  -- id makes the order repeatable
)
SELECT d.id, d.title, d.topic,
       COALESCE(1.0 / ({rrf_k} + s.rank), 0) * {w_vec}
     + COALESCE(1.0 / ({rrf_k} + l.rank), 0) * {w_fts} AS score,
       s.rank AS vec_rank, l.rank AS fts_rank
FROM semantic s
FULL OUTER JOIN lexical l USING (id)
JOIN {tbl} d USING (id)
ORDER BY score DESC, d.id
LIMIT {k};"""


@app.route("/api/hybrid/search", methods=["POST"])
def hybrid_search():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Enter a query"}), 400

    opclass = body.get("opclass", "vector_l2_ops")
    probes  = int(body.get("probes", 32))
    k       = max(1, min(50, int(body.get("k", 10))))

    # Candidate depth is the single most consequential knob here, and the old
    # default of 50 was very nearly too small to show the point. On the query
    # "why was the payment to Sakura Logistics KK returned", the one document
    # both legs found sat at vector rank 43 and lexical rank 34 — three ranks
    # inside a 50-row list. At candidates = 30 fusion would have found no
    # agreement at all and returned a plain interleave. Default to the pool cap.
    n     = max(k, min(LIMITS["rerank_pool"], int(body.get("candidates", 200))))
    rrf_k = max(1, int(body.get("rrf_k", 60)))
    w_vec = float(body.get("w_vec", 1.0))
    w_fts = float(body.get("w_fts", 1.0))
    lex   = body.get("lexical", "tsvector")      # tsvector | bm25

    if opclass not in OPCLASSES:
        return jsonify({"ok": False, "error": "Unknown opclass"}), 400
    if lex not in ("tsvector", "bm25"):
        return jsonify({"ok": False,
                        "error": "lexical must be 'tsvector' or 'bm25'"}), 400
    bm = bm25_available() if lex == "bm25" else None
    if lex == "bm25" and not bm["usable"]:
        return jsonify({"ok": False, "error": bm["reason"], "bm25": bm}), 400

    op  = OPCLASSES[opclass]["op"]
    dim = table_dim()
    qv  = vec_literal(embed_text(text, dim))
    tbl = qualified(TABLE)
    # Naming the index explicitly is required for an expression index; the
    # implicit `col <@> 'text'` form cannot resolve one.
    bq  = f"{BM25_INDEX}:{text}" if lex == "bm25" else None

    # ── Tie-breaking, and where it is safe ───────────────────────────────────
    # ts_rank_cd is not an index-ordered operator: the plan sorts anyway, so
    # adding `, id` is free and makes the order repeatable. The vector leg and
    # the BM25 leg both sort through an ordering operator (<->, <@>), where a
    # second sort key would defeat the index scan — an ivfplus scan or BM25
    # Block-Max WAND skipping is worth far more than deterministic ties. Float
    # distances effectively never tie; BM25 with real IDF ties rarely.
    lex_order = ("ts_rank_cd(tsv, query) DESC, id" if lex == "tsvector"
                 else f"{BM25_EXPR} <@> %(bq)s")

    if lex == "bm25":
        lexical_cte = f"""    lexical AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY {lex_order}) AS rank
        FROM {tbl} ORDER BY {lex_order} LIMIT %(n)s
    )"""
    else:
        lexical_cte = f"""    lexical AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY {lex_order}) AS rank
        FROM {tbl}, plainto_tsquery('english', %(q)s) query
        WHERE tsv @@ query ORDER BY {lex_order} LIMIT %(n)s
    )"""

    sql = f"""
    WITH semantic AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding {op} %(qv)s) AS rank
        FROM {tbl} ORDER BY embedding {op} %(qv)s LIMIT %(n)s
    ),
{lexical_cte}
    SELECT d.id, d.title, d.topic, d.category_id,
           COALESCE(1.0 / (%(rrf_k)s + s.rank), 0) * %(wv)s
         + COALESCE(1.0 / (%(rrf_k)s + l.rank), 0) * %(wf)s AS score,
           s.rank AS vec_rank, l.rank AS fts_rank
    FROM semantic s
    FULL OUTER JOIN lexical l USING (id)
    JOIN {tbl} d USING (id)
    ORDER BY score DESC, d.id LIMIT %(k)s"""

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SET LOCAL statement_timeout = 60000")
        cur.execute("SET LOCAL ivfplus.probes = %s", (probes,))

        # ── Each leg on its own ──────────────────────────────────────────────
        # Run at the CANDIDATE depth, not k, and with the same ORDER BY as the
        # CTE above. Previously these were separate queries at LIMIT k, which
        # meant the ranks on screen came from a different execution than the
        # ranks in the fused score — and with tied lexical scores the two
        # executions broke the ties differently. Display takes the first k.
        _vec_sql = (f"SELECT id, title, topic, category_id,\n"
                    f"       embedding {op} %s AS distance\n"
                    f"FROM {tbl} ORDER BY embedding {op} %s LIMIT %s")
        t0 = time.perf_counter()
        cur.execute(_vec_sql, (qv, qv, n))
        vec_all = [dict(r) for r in cur.fetchall()]
        vec_ms = (time.perf_counter() - t0) * 1000

        if lex == "bm25":
            _fts_sql = (f"SELECT id, title, topic, category_id,\n"
                        f"       {BM25_EXPR} <@> %s AS rank\n"
                        f"FROM {tbl} ORDER BY {BM25_EXPR} <@> %s LIMIT %s")
            _fts_args = (bq, bq, n)
        else:
            _fts_sql = (f"SELECT id, title, topic, category_id,\n"
                        f"       ts_rank_cd(tsv, query) AS rank\n"
                        f"FROM {tbl}, plainto_tsquery('english', %s) query\n"
                        f"WHERE tsv @@ query\n"
                        f"ORDER BY ts_rank_cd(tsv, query) DESC, id LIMIT %s")
            _fts_args = (text, n)
        t0 = time.perf_counter()
        cur.execute(_fts_sql, _fts_args)
        fts_all = [dict(r) for r in cur.fetchall()]
        fts_ms = (time.perf_counter() - t0) * 1000

        # ── Fusion ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        cur.execute(sql, {"qv": qv, "q": text, "bq": bq, "n": n, "rrf_k": rrf_k,
                          "wv": w_vec, "wf": w_fts, "k": k})
        fused = [dict(r) for r in cur.fetchall()]
        fused_ms = (time.perf_counter() - t0) * 1000
        conn.commit()

        log_query("hybrid-vector", _vec_sql,
                  settings=[f"ivfplus.probes = {probes}"], ms=vec_ms,
                  rows=len(vec_all), label=text,
                  sql_full=resolve_sql(cur, _vec_sql, (qv, qv, n)))
        log_query(f"hybrid-lexical-{lex}", _fts_sql, ms=fts_ms,
                  rows=len(fts_all), label=text,
                  sql_full=resolve_sql(cur, _fts_sql, _fts_args))
        log_query("hybrid-rrf", sql, settings=[f"ivfplus.probes = {probes}"],
                  params={"q": text, "n": n, "rrf_k": rrf_k,
                          "w_vec": w_vec, "w_fts": w_fts},
                  ms=fused_ms, rows=len(fused), label=text,
                  sql_full=resolve_sql(cur, sql,
                      {"qv": qv, "q": text, "n": n, "rrf_k": rrf_k,
                       "wv": w_vec, "wf": w_fts, "k": k}))

        for r in vec_all:
            r["distance"] = round(float(r["distance"]), 6)
        for r in fts_all:
            r["rank"] = round(float(r["rank"]), 6)
        for r in fused:
            r["score"] = round(float(r["score"]), 6)

        # ── Overlap, measured where fusion measured it ───────────────────────
        # The old code intersected the two top-k lists, so a document the legs
        # agreed on at ranks 43 and 34 counted as no overlap at all — while the
        # fused list had promoted exactly that document to first place.
        vec_ids = [r["id"] for r in vec_all]
        fts_ids = [r["id"] for r in fts_all]
        vec_set, fts_set = set(vec_ids), set(fts_ids)
        overlap = sorted(vec_set & fts_set)
        overlap_topk = sorted(set(vec_ids[:k]) & set(fts_ids[:k]))

        # Rank positions of the agreements, so the UI can say where they were.
        vec_pos = {i: p + 1 for p, i in enumerate(vec_ids)}
        fts_pos = {i: p + 1 for p, i in enumerate(fts_ids)}
        agreements = sorted(
            ({"id": i, "vec_rank": vec_pos[i], "fts_rank": fts_pos[i],
              "score": round(w_vec / (rrf_k + vec_pos[i])
                             + w_fts / (rrf_k + fts_pos[i]), 6)}
             for i in overlap),
            key=lambda a: -a["score"])[:10]

        # Does the fused list agree with the legs about the ranks? If the two
        # ever disagree, ties are still being broken non-deterministically and
        # the numbers on screen cannot be trusted. Surface it rather than hide.
        mismatch = sum(1 for r in fused
                       if (r.get("vec_rank") or None) != vec_pos.get(r["id"])
                       or (r.get("fts_rank") or None) != fts_pos.get(r["id"]))

        # Do the lexical scores discriminate at all, or is the order a coin toss?
        lex_top = [r["rank"] for r in fts_all[:k]]
        lex_tied = len(lex_top) > 1 and len(set(lex_top)) == 1

        pretty = HYBRID_SQL_TEMPLATE.format(op=op, tbl=tbl, n=n, rrf_k=rrf_k,
                                            w_vec=w_vec, w_fts=w_fts, k=k)

        notes = []
        if not overlap:
            notes.append(
                f"Neither leg returned a single document the other also "
                f"returned, anywhere in {n} candidates. RRF has nothing to "
                f"fuse, so the fused list is just the two lists interleaved by "
                f"rank — every score below is 1/({rrf_k}+rank) from one leg. "
                f"That is a real finding about this query, not a failure.")
        else:
            best = agreements[0]
            notes.append(
                f"{len(overlap)} document(s) were found by both legs. The "
                f"strongest was ranked {best['vec_rank']} by the vector leg and "
                f"{best['fts_rank']} by the lexical leg — a document only one "
                f"leg finds cannot score above "
                f"{round(max(w_vec, w_fts) / (rrf_k + 1), 5)}, so agreement is "
                f"the only thing that can reach the top of this list.")
        if lex_tied:
            notes.append(
                "Every lexical result in the top " + str(k) + " carries the "
                "same score, so their order is an arbitrary tie-break and the "
                "ranks feeding RRF from that leg are not meaningful. This is "
                "the synthetic corpus — uniform document lengths and flat IDF "
                "— not the ranker. " +
                ("Switch the lexical engine to pg_textsearch: BM25 has IDF, "
                 "which ts_rank_cd does not." if lex == "tsvector" else
                 "Real documents are needed before any ranking claim."))
        if mismatch:
            notes.append(
                f"{mismatch} of the fused rows disagree with the per-leg ranks "
                f"shown alongside them. Ties are still being resolved "
                f"differently between executions — treat the ranks as "
                f"indicative only.")
        if not fts_all:
            missing = terms_absent_from_corpus(text)
            if missing:
                notes.append(
                    "The lexical leg matched nothing because these words do not "
                    "occur anywhere in the loaded corpus: " + ", ".join(missing)
                    + ". plainto_tsquery ANDs its terms, so one absent word is "
                    "enough to return zero rows. Pick a query from the chips "
                    f"above, which are drawn from the '{CORPUS_NAME}' corpus.")
            else:
                notes.append("The lexical leg matched nothing — every term "
                             "exists in the corpus, but no single document "
                             "contains all of them. plainto_tsquery ANDs its "
                             "terms.")

        return jsonify({
            "ok": True, "query": text, "probes": probes, "lexical_engine": lex,
            "bm25": bm, "k": k, "candidates": n, "rrf_k": rrf_k,
            "vector":  {"rows": vec_all[:k], "ms": round(vec_ms, 2),
                        "returned": len(vec_all)},
            "lexical": {"rows": fts_all[:k], "ms": round(fts_ms, 2),
                        "returned": len(fts_all)},
            "fused":   {"rows": fused, "ms": round(fused_ms, 2)},
            "overlap": overlap,                 # across all n candidates
            "overlap_topk": overlap_topk,       # within the displayed top k
            "agreements": agreements,
            "vector_only":  sorted(vec_set - fts_set),
            "lexical_only": sorted(fts_set - vec_set),
            "lexical_tied": lex_tied,
            "rank_mismatch": mismatch,
            "sql": pretty, "notes": notes,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  FILTERING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/filter/demo", methods=["POST"])
def filter_demo():
    """Same query, filtered and unfiltered, to show the post-scan shortfall."""
    body = request.get_json(silent=True) or {}
    try:
        qv, label = resolve_query_vector(body)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    opclass = body.get("opclass", "vector_l2_ops")
    probes  = int(body.get("probes", 10))
    limit   = int(body.get("limit", 10))
    cat     = body.get("category_id")
    where   = f"category_id = {int(cat)}" if cat not in (None, "", "any") else None

    try:
        plain = run_search(qv, opclass, probes, limit, log_kind="filter-unfiltered")
        filt  = (run_search(qv, opclass, probes, limit, where=where, explain=True,
                            log_kind="filter-filtered") if where else None)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    out = {"ok": True, "label": label, "probes": probes, "limit": limit,
           "unfiltered": plain, "filtered": filt, "where": where}
    if filt:
        out["shortfall"] = limit - len(filt["rows"])
        out["note"] = ("Filtering is applied after the index scan, as with pgvector's "
                       "IVFFlat, so a selective filter can return fewer than LIMIT rows. "
                       "Fix it with a partial index or a wider probes.")
    return jsonify(out)


@app.route("/api/filter/partial_index", methods=["POST"])
def filter_partial_index():
    body    = request.get_json(silent=True) or {}
    cat     = int(body.get("category_id", 2))
    lists   = int(body.get("lists", 100))
    opclass = body.get("opclass", "vector_l2_ops")
    name    = f"{IDX_PREFIX}_cat{cat}"
    sql     = (f'CREATE INDEX IF NOT EXISTS "{name}" ON {qualified(TABLE)}\n'
               f"    USING ivfplus (embedding {opclass})\n"
               f"    WITH (lists = {lists})\n"
               f"    WHERE (category_id = {cat})")
    # A partial index only clusters the rows its WHERE clause matches, so the
    # k-means sample — and therefore the build memory — is bounded by that
    # subset, not by the whole table. Sizing off the full row count would
    # massively over-reserve.
    dim = table_dim()
    matching = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)} "
                          f"WHERE category_id = %s", (cat,), default=0) or 0)
    if matching == 0:
        return jsonify({"ok": False,
                        "error": f"No rows with category_id = {cat}, so there is "
                                 f"nothing for the build to cluster."}), 400

    already = bool(scalar("SELECT to_regclass(%s) IS NOT NULL",
                          (f"{SCHEMA}.{name}",), default=False))
    if already:
        # CREATE INDEX IF NOT EXISTS would return instantly and look like a
        # suspiciously fast build. Say so instead.
        return jsonify({"ok": True, "name": name, "seconds": 0, "sql": sql,
                        "already_existed": True, "matching_rows": matching,
                        "notices": [f'Index "{name}" already exists — nothing was '
                                    f'rebuilt. Drop it first to time a fresh build.']})

    mem      = ivf_build_memory(matching, dim, lists)
    server_b = parse_mem_setting(
        scalar("SELECT current_setting('maintenance_work_mem')")) or 0
    cap_b    = parse_mem_setting(BUILD_MAX_MAINT_MEM) or (4 * 1024 ** 3)
    maint_mem = None
    if mem["bytes"] > server_b:
        maint_mem = fmt_mb(min(cap_b, ((mem["bytes"] + 67_108_863) // 67_108_864)
                               * 67_108_864))

    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        required_mb = None
        for attempt in (1, 2):
            if maint_mem and _MWM_RE.match(maint_mem):
                cur.execute("SET maintenance_work_mem = %s", (maint_mem,))
            t0 = time.perf_counter()
            try:
                cur.execute(sql)
                el = round(time.perf_counter() - t0, 3)
                log_query("partial-index", sql,
                          settings=[f"maintenance_work_mem = {maint_mem}"] if maint_mem else None,
                          ms=el * 1000, rows=matching, label=name)
                break
            except Exception as e:
                hit = MEM_ERROR_RE.search(str(e))
                if hit and attempt == 1:
                    required_mb = int(hit.group(1))
                    need = int(math.ceil(required_mb * 1.1 * 1024 * 1024
                                         / 67_108_864)) * 67_108_864
                    if need > cap_b:
                        raise
                    maint_mem = fmt_mb(need)
                    continue
                raise
        return jsonify({"ok": True, "name": name, "seconds": el, "sql": sql,
                        "matching_rows": matching,
                        "maintenance_work_mem": maint_mem,
                        "mem_estimated_mb": mem["mb"],
                        "mem_required_mb": required_mb,
                        "notices": clean_notices(conn.notices)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "sql": sql}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAINTENANCE — churn, health, REINDEX, rerank stats
# ══════════════════════════════════════════════════════════════════════════════

def _churn_worker(inserts, updates, deletes, dim):
    CHURN_STATE.update({"running": True, "phase": "starting", "done": 0,
                        "total": inserts + updates + deletes, "error": None})
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        rng = np.random.default_rng(int(time.time()) & 0xFFFF)

        if inserts:
            CHURN_STATE["phase"] = "insert"
            step = 1000
            left = inserts
            pyrng = random.Random(int(time.time()) & 0xFFFF)
            while left > 0:
                n = min(step, left)
                copy_rows(cur, make_batch(rng, dim, n, pyrng=pyrng),
                          binary=COPY_BINARY_DEFAULT, log_kind="churn-insert")
                left -= n
                CHURN_STATE["done"] += n

        if updates:
            CHURN_STATE["phase"] = "update"
            _u = (f"UPDATE {qualified(TABLE)} SET body = body || ' revised' "
                  f"WHERE id IN (SELECT id FROM {qualified(TABLE)} "
                  f"ORDER BY random() LIMIT {updates})")
            _t = time.perf_counter()
            cur.execute(_u)
            log_query("churn-update", _u, ms=(time.perf_counter() - _t) * 1000,
                      rows=cur.rowcount)
            CHURN_STATE["done"] += updates

        if deletes:
            CHURN_STATE["phase"] = "delete"
            _d = (f"DELETE FROM {qualified(TABLE)} WHERE id IN "
                  f"(SELECT id FROM {qualified(TABLE)} ORDER BY random() LIMIT {deletes})")
            _t = time.perf_counter()
            cur.execute(_d)
            log_query("churn-delete", _d, ms=(time.perf_counter() - _t) * 1000,
                      rows=cur.rowcount)
            CHURN_STATE["done"] += deletes
            CHURN_STATE["phase"] = "vacuum"
            cur.execute(f"VACUUM {qualified(TABLE)}")

        CHURN_STATE["phase"] = "done"
    except Exception as e:
        CHURN_STATE["error"] = str(e)
        CHURN_STATE["phase"] = "error"
    finally:
        CHURN_STATE["running"] = False
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/maint/churn", methods=["POST"])
def maint_churn():
    """
    Drift the table away from the fixed centroids on purpose. INSERT lands in the
    append region (slower bit-sliced scan), UPDATE is delete+insert so the row
    moves there too, DELETE/VACUUM tombstones a lane that is never reused.
    """
    if CHURN_STATE["running"]:
        return jsonify({"ok": False, "error": "Churn already running"}), 400
    body = request.get_json(silent=True) or {}
    ins  = max(0, min(500000, int(body.get("inserts", 5000))))
    upd  = max(0, min(500000, int(body.get("updates", 2000))))
    dele = max(0, min(500000, int(body.get("deletes", 1000))))
    dim  = table_dim()
    if not dim:
        return jsonify({"ok": False, "error": "No table"}), 400
    threading.Thread(target=_churn_worker, args=(ins, upd, dele, dim), daemon=True).start()
    return jsonify({"ok": True, "inserts": ins, "updates": upd, "deletes": dele})


@app.route("/api/maint/churn/status")
def maint_churn_status():
    return jsonify(CHURN_STATE)


@app.route("/api/maint/health")
def maint_health():
    name = request.args.get("name")
    if name:
        return jsonify({"ok": True, "rows": [dict(index_health(name), name=name)]})
    out = []
    for r in ivfplus_indexes():
        if r["am"] == "ivfplus":
            out.append(dict(index_health(r["name"]), name=r["name"], size=r["size"]))
    return jsonify({"ok": True, "rows": out})


@app.route("/api/maint/reindex", methods=["POST"])
def maint_reindex():
    """The only operation that repacks append-region rows and reclaims dead lanes."""
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    conc = bool(body.get("concurrently", True))
    if not name or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        return jsonify({"ok": False, "error": "Bad index name"}), 400

    before = index_health(name)
    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        sql = f'REINDEX INDEX {"CONCURRENTLY " if conc else ""}"{SCHEMA}"."{name}"'
        t0 = time.perf_counter()
        cur.execute(sql)
        el = round(time.perf_counter() - t0, 3)
        log_query("reindex", sql, ms=el * 1000, label=name)
        after = index_health(name)
        return jsonify({"ok": True, "seconds": el, "sql": sql,
                        "before": before, "after": after,
                        "notices": clean_notices(conn.notices)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/maint/rerank_stats")
def maint_rerank_stats():
    """
    Backend-process counters only: they start at zero when the backend starts,
    accumulate across every query and every ivfplus index that process touches,
    and vanish with the connection. Since this API opens a fresh connection per
    request, a read here is effectively always zero — which is itself the lesson.
    """
    avail = scalar("SELECT edb_vectorplus_rerank_policy_available()", default=None)
    if not avail:
        return jsonify({"ok": True, "available": False,
                        "note": "This server is stock PostgreSQL: the f16 rerank copy is "
                                "always present, store_vectors does not exist, and the "
                                "policy functions are not installed. A postgres-im-rerank "
                                "build drops the copy and reranks in the executor against "
                                "the heap."})
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Exercise the policy on this same connection so the counters are non-zero.
        dim = table_dim()
        qv  = vec_literal(embed_text("vector index probe", dim))
        cur.execute("SET LOCAL ivfplus.probes = 16")
        cur.execute(f"SELECT id FROM {qualified(TABLE)} ORDER BY embedding <-> %s LIMIT 10",
                    (qv,))
        cur.fetchall()
        cur.execute("SELECT * FROM edb_vectorplus_rerank_stats(false)")
        row = cur.fetchone()
        conn.commit()
        return jsonify({"ok": True, "available": True, "stats": dict(row) if row else {},
                        "note": "Counters are per backend process and not per index. They "
                                "reset with the connection."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  SQL EDITOR
# ══════════════════════════════════════════════════════════════════════════════

def sql_presets():
    t = qualified(TABLE)
    return [
        {"label": "extension versions",
         "sql": "SELECT extname, extversion FROM pg_extension "
                "WHERE extname IN ('edb_vectorplus','vector');"},
        {"label": "rerank policy?",
         "sql": "SELECT edb_vectorplus_rerank_policy_available();"},
        {"label": "index health (all ivfplus)",
         "sql": "SELECT i.relname,\n"
                "       (check_ivfplus_index_health(i.oid)).*\n"
                "FROM pg_class i JOIN pg_am am ON am.oid = i.relam\n"
                "WHERE am.amname = 'ivfplus';"},
        {"label": "index sizes",
         "sql": "SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size\n"
                f"FROM pg_stat_user_indexes WHERE relname = '{TABLE}';"},
        {"label": "opclasses",
         "sql": "SELECT opcname, opcdefault FROM pg_opclass o JOIN pg_am a ON a.oid = o.opcmethod\n"
                "WHERE a.amname = 'ivfplus' ORDER BY opcname;"},
        {"label": "row count by topic",
         "sql": f"SELECT topic, count(*) FROM {t} GROUP BY topic ORDER BY 2 DESC;"},
        {"label": "build settings",
         "sql": "SELECT name, setting, unit FROM pg_settings\n"
                "WHERE name IN ('maintenance_work_mem','max_parallel_maintenance_workers',\n"
                "               'block_size','shared_buffers');"},
        {"label": "ivfplus GUCs",
         "sql": "SELECT name, setting, min_val, max_val FROM pg_settings\n"
                "WHERE name LIKE 'ivfplus.%';"},
        {"label": "build progress",
         "sql": "SELECT pid, phase, blocks_done, blocks_total, tuples_done\n"
                "FROM pg_stat_progress_create_index;"},
    ]


@app.route("/api/sqleditor/presets")
def sqleditor_presets():
    return jsonify(sql_presets())


@app.route("/api/sqleditor/query", methods=["POST"])
def sqleditor_query():
    body    = request.get_json(silent=True) or {}
    sql     = (body.get("sql") or "").strip()
    confirm = bool(body.get("confirm"))
    if not sql:
        return jsonify({"ok": False, "error": "Empty query"}), 400
    if ";" in sql.rstrip().rstrip(";"):
        return jsonify({"ok": False, "error": "Only a single statement per run"}), 400
    if DESTRUCTIVE_SQL_RE.search(sql) and not confirm:
        return jsonify({"ok": False, "needs_confirm": True,
                        "error": "This looks like a data-changing statement. "
                                 "Resend with confirm:true to proceed."}), 400

    conn = None
    t0 = time.perf_counter()
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '120s'")
        try:
            cur.execute(sql)
        except Exception as e:
            log_query("sql-editor", sql, error=e)
            raise
        elapsed = round(time.perf_counter() - t0, 3)
        notices = clean_notices(conn.notices)
        log_query("sql-editor", sql, ms=elapsed * 1000, sql_full=sql,
                  rows=(cur.rowcount if not cur.description else None))
        if cur.description:
            cols = [d[0] for d in cur.description]
            rows = [[str(v) if v is not None else None for v in r] for r in cur.fetchmany(500)]
            return jsonify({"ok": True, "columns": cols, "rows": rows,
                            "row_count": len(rows), "truncated": len(rows) == 500,
                            "elapsed": elapsed, "notices": notices})
        return jsonify({"ok": True, "columns": [], "rows": [],
                        "affected": cur.rowcount, "elapsed": elapsed, "notices": notices})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass



# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY & CROSS-ENGINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/history/builds")
def history_builds():
    limit = max(1, min(500, int(request.args.get("limit", 100))))
    rows = store_query("SELECT * FROM builds ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["size"] = human_bytes(r.get("index_bytes"))
    return jsonify({"ok": True, "rows": rows, "target": TARGET_NAME,
                    "results_db": RESULTS_DB})


@app.route("/api/history/recalls")
def history_recalls():
    limit = max(1, min(2000, int(request.args.get("limit", 500))))
    rows = store_query("SELECT * FROM recalls ORDER BY id DESC LIMIT ?", (limit,))
    return jsonify({"ok": True, "rows": rows})


@app.route("/api/history/compare")
def history_compare():
    """
    Three comparisons over everything ever recorded, including runs the vpbench
    CLI made against other databases and engines.
    """
    floor = float(request.args.get("floor", 0.95))

    # Latest build per (target, engine, parameter set), baselined on the fastest.
    builds = store_query("""
        SELECT * FROM builds b WHERE id = (
            SELECT max(id) FROM builds x
             WHERE x.target = b.target AND x.engine = b.engine
               AND ifnull(x.lists,-1) = ifnull(b.lists,-1)
               AND ifnull(x.m,-1)     = ifnull(b.m,-1))
        ORDER BY rows, engine, lists, m""")
    fastest = min((b["build_seconds"] for b in builds if b["build_seconds"]), default=None)
    for b in builds:
        b["size"] = human_bytes(b.get("index_bytes"))
        b["rel"] = (round(b["build_seconds"] / fastest, 2)
                    if fastest and b.get("build_seconds") else None)

    # The question that matters: what latency do you pay for a given recall?
    # Among settings that clear the floor, take the cheapest — not the newest.
    recalls = store_query("""
        SELECT r.* FROM recalls r
        JOIN (SELECT target, engine, k, min(ms) AS best
                FROM recalls WHERE recall >= ?
               GROUP BY target, engine, k) g
          ON g.target = r.target AND g.engine = r.engine
         AND g.k = r.k AND g.best = r.ms
        WHERE r.recall >= ?
        GROUP BY r.target, r.engine, r.k
        ORDER BY r.ms""", (floor, floor))

    # How well the build-memory model is tracking what the server actually wants.
    memory = store_query("""SELECT * FROM builds
                            WHERE mem_required_mb IS NOT NULL
                            ORDER BY id DESC LIMIT 30""")
    for m_ in memory:
        est, req = m_.get("mem_estimated_mb"), m_.get("mem_required_mb")
        m_["error_pct"] = round(abs(est - req) / req * 100, 1) if est and req else None

    curves = store_query("""SELECT target, label, engine, k, param, value, recall, ms
                            FROM recalls ORDER BY target, engine, k, value""")
    return jsonify({"ok": True, "floor": floor, "builds": builds,
                    "recalls": recalls, "memory": memory, "curves": curves})


@app.route("/api/history/delete", methods=["POST"])
def history_delete():
    body = request.get_json(silent=True) or {}
    what, rid = body.get("table"), body.get("id")
    if what not in ("builds", "recalls"):
        return jsonify({"ok": False, "error": "table must be builds or recalls"}), 400
    try:
        with _STORE_LOCK:
            c = store_conn()
            if rid:
                c.execute(f"DELETE FROM {what} WHERE id = ?", (int(rid),))
            elif body.get("all"):
                c.execute(f"DELETE FROM {what}")
            else:
                c.close()
                return jsonify({"ok": False, "error": "pass id, or all:true"}), 400
            c.commit(); c.close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400



@app.route("/api/log/queries")
def log_queries():
    """The recorded statements, newest first, optionally filtered by kind."""
    limit = max(1, min(1000, int(request.args.get("limit", 200))))
    kind  = request.args.get("kind") or None
    if kind and kind != "all":
        rows = store_query("SELECT * FROM queries WHERE kind = ? "
                           "ORDER BY id DESC LIMIT ?", (kind, limit))
    else:
        rows = store_query("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,))
    for r in rows:
        r["has_full"] = bool(r.get("sql_full"))
        # The resolved form carries the embedding and can be ~7 kB. Keep it out
        # of the list response; /api/log/query/<id> serves it on demand.
        r["sql_full"] = None
    kinds = [r["kind"] for r in
             store_query("SELECT kind, count(*) AS n FROM queries "
                         "GROUP BY kind ORDER BY n DESC")]
    return jsonify({"ok": True, "rows": rows, "kinds": kinds,
                    "log_file": LOG_FILE, "results_db": RESULTS_DB})


@app.route("/api/log/file")
def log_file_tail():
    n = max(1, min(2000, int(request.args.get("lines", 200))))
    try:
        with open(LOG_FILE, "r", errors="replace") as fh:
            tail = fh.readlines()[-n:]
        return jsonify({"ok": True, "path": LOG_FILE, "text": "".join(tail)})
    except FileNotFoundError:
        return jsonify({"ok": True, "path": LOG_FILE,
                        "text": "(nothing logged yet)"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/log/query/<int:qid>")
def log_query_one(qid):
    """One statement in full, parameters substituted — runnable in psql as-is."""
    rows = store_query("SELECT * FROM queries WHERE id = ?", (qid,))
    if not rows:
        return jsonify({"ok": False, "error": f"No logged query #{qid}"}), 404
    r = rows[0]
    body = r.get("sql_full") or r.get("sql") or ""
    if r.get("settings"):
        body = "\n".join(f"SET {x};" for x in r["settings"].split("; ")) + "\n\n" + body
    if body.strip() and not body.rstrip().endswith(";"):
        body = body.rstrip() + ";"
    return jsonify({"ok": True, "row": r, "runnable": body,
                    "resolved": bool(r.get("sql_full")),
                    "bytes": len(body)})


@app.route("/api/log/level", methods=["GET", "POST"])
def log_level():
    """
    'user' records the statements you asked for. 'all' adds the app's own
    housekeeping — status polls, health checks, row counts. Complete, but the
    status poll alone runs about a dozen statements every four seconds, so it
    is off by default rather than on.
    """
    global LOG_LEVEL
    if request.method == "POST":
        want = (request.get_json(silent=True) or {}).get("level", "user")
        if want not in ("user", "all"):
            return jsonify({"ok": False, "error": "level must be 'user' or 'all'"}), 400
        LOG_LEVEL = want
        _qlog.info(f"[log] level set to {LOG_LEVEL}")
    counts = store_query("SELECT kind, count(*) AS n FROM queries "
                         "GROUP BY kind ORDER BY n DESC")
    return jsonify({"ok": True, "level": LOG_LEVEL, "counts": counts})


@app.route("/api/log/clear", methods=["POST"])
def log_clear():
    try:
        with _STORE_LOCK:
            c = store_conn(); c.execute("DELETE FROM queries"); c.commit(); c.close()
        if (request.get_json(silent=True) or {}).get("file"):
            open(LOG_FILE, "w").close()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400



# ══════════════════════════════════════════════════════════════════════════════
#  BM25  (pg_textsearch)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/bm25/status")
def bm25_status():
    try:
        st = bm25_available()
    except Exception as e:
        return jsonify({"ok": False, "usable": False, "extension": None,
                        "am": False, "preloaded": False, "pg_major": 0,
                        "index": None, "reason": f"cannot reach the database: {e}"})
    st = dict(st)
    st["expr"] = BM25_EXPR
    st["index_name"] = BM25_INDEX
    try:
        st["gucs"] = q("SELECT name, setting FROM pg_settings "
                       "WHERE name LIKE 'pg_textsearch.%' ORDER BY name")
    except Exception:
        st["gucs"] = []
    st["install_hint"] = (
        "pg_textsearch must be in shared_preload_libraries and the server "
        "restarted; it supports PostgreSQL 17 and 18 only.")
    return jsonify({"ok": True, **st})


@app.route("/api/bm25/build", methods=["POST"])
def bm25_build():
    """
    Build the BM25 index over the same text the tsvector column covers, so the
    two rankers are compared on identical input.

    An expression index avoids touching the table: no new column, no reload.
    The cost is that queries must repeat the expression and name the index —
    the implicit `col <@> 'text'` form cannot resolve an expression index.
    """
    body   = request.get_json(silent=True) or {}
    cfg    = body.get("text_config", "english")
    k1     = float(body.get("k1", 1.2))
    b      = float(body.get("b", 0.75))
    merge  = bool(body.get("force_merge", True))
    if not re.fullmatch(r"[a-z_]+", cfg):
        return jsonify({"ok": False, "error": "Bad text_config"}), 400
    if not (0.1 <= k1 <= 10.0) or not (0.0 <= b <= 1.0):
        return jsonify({"ok": False, "error": "k1 must be 0.1–10.0 and b 0.0–1.0"}), 400

    st = bm25_available()
    if not st["extension"]:
        return jsonify({"ok": False, "error": st["reason"], "status": st}), 400

    rows = int(scalar(f"SELECT count(*) FROM {qualified(TABLE)}", default=0) or 0)
    if not rows:
        return jsonify({"ok": False, "error": "Load data before building the index — "
                                              "pg_textsearch is faster built after the "
                                              "load, like every other index type."}), 400

    sql = (f'CREATE INDEX "{BM25_INDEX}" ON {qualified(TABLE)}\n'
           f"    USING bm25 ({BM25_EXPR})\n"
           f"    WITH (text_config = '{cfg}', k1 = {k1}, b = {b})")

    conn = None
    try:
        conn = get_conn(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 0")
        # Parallel builds need at least 64MB or the planner silently drops to
        # serial. Nothing reports that, so make sure of it here.
        mw = parse_mem_setting(scalar("SELECT current_setting('maintenance_work_mem')")) or 0
        if mw < 64 * 1024 * 1024:
            cur.execute("SET maintenance_work_mem = '256MB'")
        cur.execute(f'DROP INDEX IF EXISTS "{SCHEMA}"."{BM25_INDEX}"')

        t0 = time.perf_counter()
        cur.execute(sql)
        build_s = time.perf_counter() - t0
        notices = clean_notices(conn.notices)
        log_query("bm25-build", sql, ms=build_s * 1000, rows=rows, label=BM25_INDEX)

        merge_s = None
        if merge:
            # Analogous to Lucene's forceMerge(1): collapse the LSM segments
            # into one so query time is not paying for a segment-per-level scan
            # left over from the bulk load.
            t1 = time.perf_counter()
            cur.execute("SELECT bm25_force_merge(%s)", (BM25_INDEX,))
            merge_s = time.perf_counter() - t1
            notices += clean_notices(conn.notices)

        size_b = scalar("SELECT pg_relation_size(%s::regclass)",
                        (f"{SCHEMA}.{BM25_INDEX}",), default=0) or 0
        gin_b = scalar("""SELECT pg_relation_size(i.oid) FROM pg_class i
                          JOIN pg_index x ON x.indexrelid = i.oid
                          JOIN pg_class t ON t.oid = x.indrelid
                          JOIN pg_am am ON am.oid = i.relam
                          WHERE t.relname = %s AND am.amname = 'gin'""",
                       (TABLE,), default=0) or 0

        store_insert("builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     target=TARGET_NAME, label=TARGET_LABEL, engine="bm25", am="bm25",
                     dsn_db=DB_CONFIG.get("database"), rows=rows, dim=None,
                     opclass=cfg, index_name=BM25_INDEX,
                     build_seconds=round(build_s, 3),
                     rows_per_sec=round(rows / build_s, 1) if build_s else 0,
                     index_bytes=int(size_b),
                     bytes_per_row=round(size_b / rows, 2) if rows else 0,
                     pg_version=scalar("SELECT current_setting('server_version')"),
                     ext_version=st["extension"], notices=" | ".join(notices),
                     note=f"k1={k1} b={b} text_config={cfg}")

        return jsonify({"ok": True, "sql": sql, "seconds": round(build_s, 3),
                        "force_merge_seconds": round(merge_s, 3) if merge_s else None,
                        "rows": rows, "bytes": int(size_b),
                        "size": human_bytes(size_b),
                        "bytes_per_row": round(size_b / rows, 2) if rows else 0,
                        "gin_bytes": int(gin_b), "gin_size": human_bytes(gin_b),
                        "vs_gin": round(size_b / gin_b, 2) if gin_b else None,
                        "notices": notices})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "sql": sql}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


@app.route("/api/bm25/compare", methods=["POST"])
def bm25_compare():
    """
    The same query through both lexical rankers, over the same text.

    ts_rank_cd scores only rows that match the tsquery, using term frequency
    within the document. BM25 adds IDF, term-frequency saturation and length
    normalization from corpus-wide statistics — which is the whole reason the
    ranking differs, not just the speed.
    """
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    k    = max(1, min(100, int(body.get("k", 10))))
    if not text:
        return jsonify({"ok": False, "error": "Enter a query"}), 400
    st = bm25_available()
    tbl = qualified(TABLE)

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SET LOCAL statement_timeout = 60000")

        ts_sql = (f"SELECT id, title, topic, ts_rank_cd(tsv, query) AS score\n"
                  f"FROM {tbl}, plainto_tsquery('english', %s) query\n"
                  f"WHERE tsv @@ query\n"
                  f"ORDER BY ts_rank_cd(tsv, query) DESC LIMIT %s")
        t0 = time.perf_counter()
        cur.execute(ts_sql, (text, k))
        ts_rows = [dict(r) for r in cur.fetchall()]
        ts_ms = (time.perf_counter() - t0) * 1000
        log_query("lexcompare-tsvector", ts_sql, ms=ts_ms, rows=len(ts_rows),
                  label=text, sql_full=resolve_sql(cur, ts_sql, (text, k)))

        bm_rows, bm_ms, bm_err = [], None, None
        if st["usable"]:
            bq = f"{BM25_INDEX}:{text}"
            bm_sql = (f"SELECT id, title, topic, {BM25_EXPR} <@> %s AS score\n"
                      f"FROM {tbl}\n"
                      f"ORDER BY {BM25_EXPR} <@> %s LIMIT %s")
            try:
                t0 = time.perf_counter()
                cur.execute(bm_sql, (bq, bq, k))
                bm_rows = [dict(r) for r in cur.fetchall()]
                bm_ms = (time.perf_counter() - t0) * 1000
                log_query("lexcompare-bm25", bm_sql, ms=bm_ms, rows=len(bm_rows),
                          label=text, sql_full=resolve_sql(cur, bm_sql, (bq, bq, k)))
            except Exception as e:
                bm_err = str(e)
                conn.rollback()
        conn.commit()

        for r in ts_rows:
            r["score"] = round(float(r["score"]), 6)
        for r in bm_rows:
            # <@> is the negative BM25 score; flip it so bigger means better,
            # which is what anyone reading a relevance table expects.
            r["score"] = round(-float(r["score"]), 4)

        ts_ids = [r["id"] for r in ts_rows]
        bm_ids = [r["id"] for r in bm_rows]
        overlap = sorted(set(ts_ids) & set(bm_ids))

        # Before saying anything about ranking, check the scores actually
        # discriminate. If every returned document carries the same score, the
        # order is an arbitrary tie-break and comparing the two lists says
        # nothing about either ranker.
        ts_distinct = len({r["score"] for r in ts_rows})
        bm_distinct = len({r["score"] for r in bm_rows})
        degenerate = (len(ts_rows) > 1 and ts_distinct <= 1) or \
                     (len(bm_rows) > 1 and bm_distinct <= 1)

        notes = []
        if degenerate:
            notes.append(
                "Every returned document has the same score, so both orderings are "
                "arbitrary tie-breaks and the overlap figure is meaningless. This is a "
                "property of the demo corpus, not of the rankers: documents are ~25 "
                "words drawn from a ~24-word topic vocabulary, so document lengths "
                "barely vary, term frequencies are almost all 1, and every term has "
                "nearly the same IDF. All three levers BM25 has over ts_rank_cd are "
                "flat here. Use MS MARCO or BEIR to compare ranking quality; this "
                "corpus can only support latency and index-size claims.")
        elif bm_rows and ts_rows and ts_ids[:3] != bm_ids[:3]:
            notes.append("The two rankers disagree in the top 3, and the scores do "
                         "discriminate, so this is a real ranking difference: BM25 "
                         "adds IDF, term-frequency saturation and length "
                         "normalization, which ts_rank_cd has none of.")
        if not st["usable"]:
            notes.append(st["reason"])
        return jsonify({"ok": True, "query": text, "k": k, "bm25": st,
                        "tsvector": {"rows": ts_rows, "ms": round(ts_ms, 2)},
                        "bm25_rows": {"rows": bm_rows,
                                      "ms": round(bm_ms, 2) if bm_ms else None,
                                      "error": bm_err},
                        "overlap": overlap,
                        "overlap_pct": round(100 * len(overlap) / max(1, len(ts_ids)), 1),
                        "degenerate": degenerate,
                        "distinct_scores": {"tsvector": ts_distinct, "bm25": bm_distinct},
                        "speedup": (round(ts_ms / bm_ms, 2)
                                    if bm_ms and bm_ms > 0 else None),
                        "notes": notes})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    finally:
        if conn:
            try: conn.close()
            except Exception: pass



# ── Example queries, per corpus ──────────────────────────────────────────────
# These drive the chips under the search boxes. They must come from the corpus
# actually loaded: plainto_tsquery ANDs its terms, so a single word that does
# not occur anywhere in the corpus makes the whole lexical leg return nothing.
# The BFSI set below was chosen by measuring where the lexical and semantic
# rankings disagree, so each one demonstrates something.
CORPUS_EXAMPLES = {
    "tech": [
        "tuning recall for an approximate vector index",
        "postgres vacuum and heap tuple visibility",
        "espresso grinder burr extraction",
        "kubernetes pod scheduler taint toleration",
        "portfolio volatility hedging exposure",
        "carbon emissions glacier permafrost",
    ],
    "banking": [
        "policy exception approval threshold",
        "sanctions screening adverse media",
        "collateral margin netting termination",
        "settlement reconciliation break investigation",
        "capital liquidity ratio reporting",
        "obligor exposure limit downgrade",
    ],
    "bfsi": [
        # legs disagree on intent — BM25 finds the name, vectors find the situation
        "why was the payment to Sakura Logistics KK returned",
        # described, not named — semantic recall does the work
        "customer funds moved onward within 48 hours without commercial purpose",
        # same document type, almost no shared results
        "ISDA variation margin dispute collateral eligibility",
        # lexical wins outright — exact counterparty match
        "Kanto Heavy Industries covenant",
        "documentary credit discrepancy bill of lading",
        "outage caused by a change deployed outside the approved window",
        "beneficial ownership could not be fully verified",
        "suspicious transaction report structuring",
    ],
}


@app.route("/api/corpus/examples")
def corpus_examples():
    return jsonify({"ok": True, "corpus": CORPUS_NAME,
                    "examples": CORPUS_EXAMPLES.get(CORPUS_NAME,
                                                    CORPUS_EXAMPLES["tech"]),
                    "topics": TOPIC_NAMES})


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def root():
    return send_from_directory(str(Path(__file__).parent), "vectorplus_dashboard.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(Path(__file__).parent), filename)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

import vc_engine; vc_engine.register(app, globals())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-host",     default=DB_CONFIG["host"])
    parser.add_argument("--pg-port",     type=int, default=DB_CONFIG["port"])
    parser.add_argument("--pg-user",     default=DB_CONFIG["user"])
    parser.add_argument("--pg-password", default=DB_CONFIG["password"])
    parser.add_argument("--pg-dbname",   default=DB_CONFIG["database"])
    parser.add_argument("--schema",      default=SCHEMA)
    parser.add_argument("--app-port",    type=int, default=5066)
    parser.add_argument("--target",      default=TARGET_NAME,
                        help="name this database goes under in the results store")
    parser.add_argument("--target-label", default=TARGET_LABEL, dest="target_label")
    parser.add_argument("--results-db",  default=RESULTS_DB, dest="results_db",
                        help="SQLite file shared with vpbench.py")
    parser.add_argument("--log-file",    default=LOG_FILE, dest="log_file",
                        help="text file for the query log")
    parser.add_argument("--corpus",      default=os.environ.get("VP_CORPUS", "tech"),
                        choices=["tech", "banking", "bfsi"],
                        help="vocabulary the document generator draws from")
    parser.add_argument("--log-level",   default=LOG_LEVEL, dest="log_level",
                        choices=["user", "all"],
                        help="'user' logs statements you asked for; 'all' adds "
                             "the app's own polling and health checks")
    args = parser.parse_args()

    DB_CONFIG.update({
        "host":     args.pg_host,
        "port":     args.pg_port,
        "user":     args.pg_user,
        "password": args.pg_password,
        "database": args.pg_dbname,
    })
    SCHEMA = args.schema
    TARGET_NAME = args.target
    TARGET_LABEL = args.target_label
    RESULTS_DB = args.results_db
    LOG_FILE = args.log_file
    LOG_LEVEL = args.log_level
    select_corpus(args.corpus)
    store_conn().close()                 # create the file and schema up front
    _init_query_log()

    print("\n  EDB Vector Plus — Demo Dashboard API")
    print(f"  DB     : {args.pg_user}@{args.pg_host}:{args.pg_port}/{args.pg_dbname}")
    print(f"  Schema : {SCHEMA}")
    print(f"  Target : {TARGET_NAME} ({TARGET_LABEL})")
    print(f"  Corpus : {CORPUS_NAME} ({len(TOPICS)} topics)")
    print(f"  Results: {RESULTS_DB}")
    print(f"  Log    : {LOG_FILE} (level: {LOG_LEVEL})")
    print(f"  Open   : http://localhost:{args.app_port}/\n")

    app.run(host="0.0.0.0", port=args.app_port, debug=False, threaded=True)

