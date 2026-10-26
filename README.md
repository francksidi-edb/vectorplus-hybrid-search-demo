# VectorPlus — hybrid search and vector-engine benchmark demo

Vector search (`ivfplus`: RaBitQ + hierarchical IVF) fused with full-text search
(`tsvector` or BM25 via `pg_textsearch`) using Reciprocal Rank Fusion (RRF) —
computed in a single SQL statement, with a live dashboard for setup, index
builds, recall tuning, hybrid search, filtering, health checks, and query
history.

The same dashboard also benchmarks `ivfplus` against other vector engines on
**identical rows**: VectorChord, pgvector (HNSW and IVFFlat), the same engines
on WarehousePG, and ClickHouse (HNSW and QBit). Every sweep is scored against an
exact scan, saved, ranked, and exportable.

## What's in here

```
vectorplus-hybrid-search-demo/
├── README.md
├── requirements.txt
├── vectorplus_api.py             The Flask API (hybrid search, index builds, recall lab, ...)
├── vc_engine.py                  Add-on: multi-engine benchmark (see below). Registered by one line at the
│                                 end of vectorplus_api.py:  import vc_engine; vc_engine.register(app, globals())
├── apply_hybrid_patch.py         Splices patches/hybrid_search_patch.py into vectorplus_api.py
├── patches/
│   └── hybrid_search_patch.py    The HYBRID SEARCH section: route + fused RRF query
├── dashboard/
│   └── vectorplus_dashboard.html Live dashboard UI (see "Dashboard")
└── docs/
    └── 2026-10-23 Vector Search Benchmark Review/
        ├── ... .pptx / .pdf      23-slide review of the first six benchmark runs
        ├── build_deck.py         Rebuilds the deck from the data (needs the EDB Bento Lite skill)
        ├── bench_data.py         Every number in the deck is computed here
        └── data/                 The six vp-bench-1 exports the deck is built from
```

## Running it

```bash
pip install -r requirements.txt
python3 vectorplus_api.py --corpus bfsi --app-port 5066 \
    --pg-host localhost --pg-user postgres --pg-dbname vectorplus_demo
# then open http://<host>:5066/
```

The dashboard page is served from `dashboard/` automatically (the add-on falls back to it when
`vectorplus_dashboard.html` is not next to `vectorplus_api.py`), so nothing needs copying.

`python3 vectorplus_api.py --help` lists every flag (`--pg-*`, `--schema`, `--app-port`,
`--corpus`, `--results-db`, `--log-file`, `--log-level`, ...). The connection settings can
also be changed, tested and saved from the dashboard's **Setup** tab.

Needs PostgreSQL with `edb_vectorplus` and `vector`; `vchord`, `pg_prewarm` and
`pg_buffercache` are used when present. WarehousePG and ClickHouse are optional.

## The benchmark add-on (`vc_engine.py`)

`vectorplus_api.py` is never edited by the add-on. `vc_engine.register(app, globals())`
adds the routes below and reads the host's configuration from its globals.

**Engines, each on its own copy of the same rows** (the app checks row count, id range and an
id checksum before every comparison and refuses to compare tables that differ):

| Engine | Where | Table |
|---|---|---|
| EDB Vector Plus (`ivfplus`) | Postgres | `vp_docs` (the source) |
| VectorChord (`vchordrq`) | Postgres | `vc_docs` |
| pgvector HNSW / IVFFlat | Postgres | `pgh_docs` / `pgi_docs` |
| the same three on WarehousePG | WarehousePG | `wvp_docs`, `wpgh_docs`, `wpgi_docs` |
| ClickHouse HNSW (`vector_similarity`) | ClickHouse, over HTTP | `ch_docs` |
| ClickHouse QBit (bit planes, no index) | ClickHouse | `chq_docs`, made from `ch_docs` |

**What it adds to the dashboard**

- **Setup** — a card per connection (Postgres, WarehousePG, ClickHouse) with a live status badge, an
  explicit *Connect*, and a validation that says which layer fails (service, port, login, database,
  vector index, QBit). Optional, off by default: start / stop the ClickHouse service from the page.
- **VectorChord, pgvector, WarehousePG, ClickHouse tabs** — clone or sync the table, build the index,
  see its size and build history.
- **Engine Compare** — one query on every engine, and the *recall vs latency sweep*: a background job
  with progress, ETA and Cancel; one shared probes list for every engine; repeats; warm or cold cache;
  an exact baseline that is computed once and reused; identical churn on every table.
- **Benchmark** — every sweep is saved. Ranking by a weighted score (latency at the recall bar, p95,
  best recall, build time, index size, maintenance), charts, scaling across row counts, runs combined
  across sweeps, lowest or median latency across repeats, export / import (`vp-bench-1` JSON), and a
  self-contained HTML report.
- **Scoring** — how the points are counted, worked through for the selected run.

**How a sweep is measured.** 50 query vectors (stored rows plus small noise) are scored against an exact
sequential scan. Each engine is run at a ladder of settings (probes; HNSW and ClickHouse get
`ef_search = max(k, 4 × probes)`; QBit reads `min(probes, 32)` bits), engines interleaved, one untimed
warm-up pass then timed passes, JIT off. Recall is recall@k against the exact result; latency is each
query's median over the passes, averaged over the queries.

### Environment variables read by the add-on

| Variable | Default | Meaning |
|---|---|---|
| `VP_VC_TABLE`, `VP_PGH_TABLE`, `VP_PGI_TABLE` | `vc_docs`, `pgh_docs`, `pgi_docs` | Postgres comparison tables |
| `VP_WH_VP_TABLE`, `VP_WH_PGH_TABLE`, `VP_WH_PGI_TABLE` | `wvp_docs`, `wpgh_docs`, `wpgi_docs` | WarehousePG tables |
| `VP_CH_TABLE`, `VP_CHQ_TABLE` | `ch_docs`, `chq_docs` | ClickHouse tables |
| `VP_CONN_FILE` | `~/.vectorplus_demo_connection.json` | Where saved connections go (mode 600, outside the served folder) |
| `VP_ALLOW_SERVICE_CONTROL` | off | `1` lets the page start / stop the ClickHouse service (see Security) |
| `VP_SYSTEMCTL` | `systemctl` on `PATH` | Full path to `systemctl` |
| `VP_DROP_CACHES_CMD` | none | Optional shell command for a full OS cache drop in cold mode |

The host app's own variables (`VP_CORPUS`, `VP_SCHEMA`, `VP_RESULTS_DB`, `VP_LOG_*`, ...) are unchanged.
Results are kept in the SQLite file named by `--results-db` (`vpbench.db` by default).

### Security

The dashboard has **no login**. Saved passwords are only written if you tick the box. Starting and
stopping a system service from a web page is therefore **off** unless the app is started with
`VP_ALLOW_SERVICE_CONTROL=1`, and then it runs exactly two commands, `systemctl start|stop <unit>`
(unit name checked against a strict pattern), through `sudo -n`. Give the app's user only those two commands:

```
<app user> ALL=(root) NOPASSWD: /usr/bin/systemctl start clickhouse-server, /usr/bin/systemctl stop clickhouse-server
```

Do not expose the app on an untrusted network.

## What the patch changes

`hybrid_search_patch.py` replaces the whole `HYBRID SEARCH` section of
`vectorplus_api.py` (from its banner comment down to, but not including, the
`FILTERING` banner). Four changes, in order of how much they matter:

1. **Candidate depth defaults to the rerank-pool cap (200), not 50.** This is
   the setting that decides whether RRF fusion has anything to fuse at all —
   with too shallow a candidate list, the two legs never overlap, and fusion
   degenerates into a plain interleave.
2. **The lexical leg breaks ties on `id`**, so ranks are repeatable between
   runs.
3. **The displayed legs and the fused legs are the same queries at the same
   depth** — previously these ran as separate queries at `LIMIT k`, so the
   ranks on screen didn't always match the ranks feeding the fused score.
4. **Overlap is measured across the whole candidate set, not just the top-k**,
   so the on-screen overlap counter agrees with what fusion actually did.

The route also returns **diagnostic notes** on every search: whether the two
legs found any agreement at all, whether the lexical leg's scores are tied
(and therefore not meaningful), and whether any fused rows disagree with the
per-leg ranks shown alongside them. See the VectorPlus deck
(`VectorPlus_Hybrid_Search_Demo.pptx`) for the worked example from real query
logs.

## Applying the patch

```bash
# from the directory containing vectorplus_api.py
python3 apply_hybrid_patch.py --dry-run     # preview the diff
python3 apply_hybrid_patch.py               # apply (writes a timestamped .bak first)
python3 apply_hybrid_patch.py --revert      # restore the newest backup
```

The script refuses to write anything that doesn't parse as valid Python, and
will tell you if the target section doesn't look like the expected original
(e.g. already patched, or structurally different).

After patching, restart the API — Flask runs with `debug=False`:

```bash
pkill -f vectorplus_api.py
python3 vectorplus_api.py --corpus bfsi --app-port 5066 ...
```

## Dashboard

`dashboard/vectorplus_dashboard.html` is the full UI. Tabs: Setup, Build Vectors, Index Build, Vector
Search, Recall Lab, Hybrid Search, Filtering, Health & Maintenance, VectorChord, pgvector, WarehousePG,
ClickHouse, Engine Compare, Benchmark, Scoring, History & Compare, Query Log and a SQL Editor. It talks to the
Flask API's endpoints (including `/api/hybrid/search`, patched above) and logs every SQL statement it runs
so results are always traceable back to the exact query.

## Benchmark review

`docs/2026-10-23 Vector Search Benchmark Review/` holds a 23-slide review of the first six sweeps (1M and
2M rows, 768 dimensions): the tests and queries, the results, and what is still open. To rebuild it after
adding runs, put the new `vp-bench-1` export in its `data/` folder and run `python3 build_deck.py`
(it uses the EDB Bento Lite presentation skill).

## Next steps

- [x] Add the base `vectorplus_api.py` and a `requirements.txt`
- [ ] Add `sql/` setup scripts (corpus generation, `ivfplus` extension, demo table DDL) — mirroring the
      structure used in [`tu-india-demo`](https://github.com/francksidi-edb/tu-india-demo)
- [ ] Confirm corpus name/size defaults and document the `--corpus` flag options (currently seen: `bfsi`, `tech`)
- [ ] Extend the sweeps for ivfplus on WarehousePG and ClickHouse, and add HNSW at 2M rows (see the review deck)
