# VectorPlus — ivfplus Hybrid Search Demo

Vector search (`ivfplus`: RaBitQ + hierarchical IVF) fused with full-text search
(`tsvector` or BM25 via `pg_textsearch`) using Reciprocal Rank Fusion (RRF) —
computed in a single SQL statement, with a live dashboard for setup, index
builds, recall tuning, hybrid search, filtering, health checks, and query
history.

This repo currently holds the **hybrid-search patch and dashboard** — the
Flask API itself (`vectorplus_api.py`) lives wherever it was originally built
and should be added here before this becomes a self-contained demo repo.

## What's in here

```
vectorplus-hybrid-search-demo/
├── README.md
├── apply_hybrid_patch.py         Splices patches/hybrid_search_patch.py into vectorplus_api.py
├── patches/
│   └── hybrid_search_patch.py    The HYBRID SEARCH section: route + fused RRF query
└── dashboard/
    └── vectorplus_dashboard.html Live dashboard UI (11 tabs — see below)
```

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

`dashboard/vectorplus_dashboard.html` is the full UI: Setup, Build Vectors,
Index Build, Vector Search, Recall Lab, Hybrid Search, Filtering, Health &
Maintenance, History & Compare, Query Log, and a SQL Editor. It talks to the
Flask API's endpoints (including `/api/hybrid/search`, patched above) and logs
every SQL statement it runs so results are always traceable back to the exact
query.

## Next steps to make this a complete, cloneable repo

- [ ] Add the base `vectorplus_api.py` (pre-patch) so `apply_hybrid_patch.py`
      has something to run against out of the box
- [ ] Add `sql/` setup scripts (corpus generation, `ivfplus` extension,
      demo table DDL) — mirroring the structure used in
      [`tu-india-demo`](https://github.com/francksidi-edb/tu-india-demo)
- [ ] Add a `requirements.txt` (`flask`, `flask-cors`, `psycopg2-binary`)
- [ ] Confirm corpus name/size defaults and document the `--corpus` flag
      options (currently seen: `bfsi`)
