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
