#!/usr/bin/env python3
"""
vc_engine.py — VectorChord (vchordrq) + cross-engine comparison + UI-driven
connection settings, as an add-on to vectorplus_api.py.

Hook it in with ONE line, placed above the `if __name__ == "__main__":` block
of vectorplus_api.py:

    import vc_engine; vc_engine.register(app, globals())

Why it takes globals() instead of `import vectorplus_api`: when the app is run
as `python3 vectorplus_api.py` the module is called __main__, and importing it
by name would load a second copy with its own DB_CONFIG, GEN_STATE and so on.
Sharing the live globals dict means a connection changed from the UI changes
the connection every existing tab uses, with no other edits.

Row-for-row clones of one table, one engine each:
    vp_docs  — the existing table, indexed with ivfplus   (edb_vectorplus)
    vc_docs  — indexed with vchordrq                      (vchord)
    pgh_docs — indexed with pgvector HNSW                 (vector)
    pgi_docs — indexed with pgvector IVFFlat              (vector)
Same ids, same embeddings, same text. Every comparison measures the access
method, not a difference in data.
"""

import io
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2
from flask import jsonify, request

VC_TABLE  = os.environ.get("VP_VC_TABLE", "vc_docs")
VC_PREFIX = f"{VC_TABLE}_emb"
VC_TARGET = "vectorchord"       # target name in the shared results store
VC_LABEL  = "VectorChord"
# Deliberately NOT next to the script: vectorplus_api.py serves every file in its
# own directory through a catch-all static route, so a credentials file there
# would be downloadable over HTTP. The home directory is not served.
CONN_FILE = os.environ.get("VP_CONN_FILE",
                           str(Path.home() / ".vectorplus_demo_connection.json"))

_IDENT_RE  = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_LISTS_RE  = re.compile(r"^\s*(\d+(\s*,\s*\d+)?)?\s*$")      # "", "1000", "1000,50000"
_PROBES_RE = re.compile(r"^[0-9]*(,[0-9]+)*$")                # vchordrq.probes is a string
_MWM_RE    = re.compile(r"^\d+\s*(kB|MB|GB|B)?$", re.IGNORECASE)

VC_BUILD = {"running": False, "sql": "", "started": None, "elapsed": 0, "error": None,
            "phase": "idle", "pid": None, "progress": None, "result": None, "notices": []}
CLONE    = {"running": False, "phase": "idle", "done": 0, "total": 0, "started": None,
            "elapsed": 0, "error": None, "result": None, "tables": [], "current": None}
CMP_CHURN = {"running": False, "phase": "idle", "done": 0, "total": 0, "error": None,
             "result": None}
CONN_META = {"source": "defaults / environment / command line"}

# pgvector's own access methods, each on its own clone of the table
PGH_TABLE  = os.environ.get("VP_PGH_TABLE", "pgh_docs")     # HNSW
PGI_TABLE  = os.environ.get("VP_PGI_TABLE", "pgi_docs")     # IVFFlat
PGV_TARGET = "pgvector"          # target name in the shared results store
PGV_BUILD = {"running": False, "sql": "", "started": None, "elapsed": 0, "error": None,
             "phase": "idle", "pid": None, "progress": None, "result": None, "notices": []}


def register(app, G):
    """Attach every route. G is the host module's globals()."""

    # ── thin wrappers over the host's helpers ───────────────────────────────
    def qual(name):            return G["qualified"](name)
    def schema():              return G["SCHEMA"]
    def vp_table():            return G["TABLE"]
    def q(*a, **k):            return G["q"](*a, **k)
    def q1(*a, **k):           return G["q1"](*a, **k)
    def scalar(*a, **k):       return G["scalar"](*a, **k)
    def get_conn(**k):         return G["get_conn"](**k)
    def log_query(*a, **k):    return G["log_query"](*a, **k)
    def resolve_sql(*a, **k):  return G["resolve_sql"](*a, **k)
    def store_insert(*a, **k): return G["store_insert"](*a, **k)
    def store_query(*a, **k):  return G["store_query"](*a, **k)
    def human_bytes(b):        return G["human_bytes"](b)
    def clean_notices(n):      return G["clean_notices"](n)
    def vec_literal(v):        return G["vec_literal"](v)

    def exists(table):
        return bool(scalar("SELECT to_regclass(%s) IS NOT NULL",
                           (f"{schema()}.{table}",), default=False))

    def dim_of(table):
        v = scalar("""SELECT a.atttypmod FROM pg_attribute a
                      JOIN pg_class c ON c.oid = a.attrelid
                      JOIN pg_namespace n ON n.oid = c.relnamespace
                      WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding'""",
                   (schema(), table), default=0)
        return int(v) if v and v > 0 else 0

    def count_rows(table):
        return int(scalar(f"SELECT count(*) FROM {qual(table)}", default=0) or 0)

    # ════════════════════════════════════════════════════════════════════════
    #  CONNECTION  (configured from the UI, optionally remembered)
    # ════════════════════════════════════════════════════════════════════════

    def conn_public():
        c = G["DB_CONFIG"]
        return {"host": c.get("host"), "port": c.get("port"), "user": c.get("user"),
                "dbname": c.get("database"), "schema": schema(),
                "has_password": bool(c.get("password")),
                "source": CONN_META["source"], "saved_file": CONN_FILE,
                "saved_exists": os.path.exists(CONN_FILE)}

    def load_saved():
        try:
            with open(CONN_FILE) as fh:
                s = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"  [conn] could not read {CONN_FILE}: {e}")
            return
        for k_src, k_dst in (("host", "host"), ("port", "port"), ("user", "user"),
                             ("password", "password"), ("dbname", "database")):
            if s.get(k_src) not in (None, ""):
                G["DB_CONFIG"][k_dst] = int(s[k_src]) if k_dst == "port" else s[k_src]
        if s.get("schema") and _IDENT_RE.match(s["schema"]):
            G["SCHEMA"] = s["schema"]
        CONN_META["source"] = f"saved file {CONN_FILE}"

    def candidate(body):
        """Validate what the UI sent. A blank password means 'keep the current one'."""
        cur = G["DB_CONFIG"]
        host = (body.get("host") or "").strip()
        user = (body.get("user") or "").strip()
        dbn  = (body.get("dbname") or "").strip()
        sch  = (body.get("schema") or "public").strip()
        if not host or not user or not dbn:
            raise ValueError("host, user and database are required")
        try:
            port = int(body.get("port") or 5432)
        except ValueError:
            raise ValueError("port must be a number")
        if not (1 <= port <= 65535):
            raise ValueError("port must be 1–65535")
        if not _IDENT_RE.match(sch):
            raise ValueError("schema must be letters, digits and underscores")
        pw = body.get("password")
        if body.get("clear_password"):
            pw = ""
        elif pw in (None, ""):
            pw = cur.get("password", "")
        return {"host": host, "port": port, "user": user, "password": pw,
                "database": dbn}, sch

    def probe(cfg, sch):
        """Connect with a candidate config and report what is installed there."""
        t0 = time.perf_counter()
        conn = psycopg2.connect(**cfg, connect_timeout=6)
        try:
            ms = (time.perf_counter() - t0) * 1000
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT current_setting('server_version'), current_database(), "
                        "current_user, current_setting('shared_preload_libraries')")
            ver, db, usr, pre = cur.fetchone()
            cur.execute("SELECT extname, extversion FROM pg_extension "
                        "WHERE extname IN ('edb_vectorplus','vchord','vector','pg_textsearch')")
            exts = {n: v for n, v in cur.fetchall()}
            cur.execute("SELECT to_regclass(%s) IS NOT NULL, to_regclass(%s) IS NOT NULL",
                        (f"{sch}.{G['TABLE']}", f"{sch}.{VC_TABLE}"))
            vp_t, vc_t = cur.fetchone()
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)", (sch,))
            sch_ok = cur.fetchone()[0]
            cur.execute("SELECT rolcreatedb, rolsuper FROM pg_roles WHERE rolname = current_user")
            can_db, sup = cur.fetchone()
            return {"server_version": ver, "database": db, "user": usr,
                    "connect_ms": round(ms, 1), "extensions": exts,
                    "preload": pre, "schema_exists": sch_ok,
                    "vp_table": vp_t, "vc_table": vc_t, "superuser": sup,
                    "vchord_preloaded": "vchord" in (pre or "")}
        finally:
            conn.close()

    def busy():
        for name, st in (("data load", G["GEN_STATE"]), ("index build", G["BUILD_STATE"]),
                         ("VectorChord build", VC_BUILD), ("pgvector build", PGV_BUILD), ("table clone", CLONE),
                         ("churn", G["CHURN_STATE"]), ("comparison churn", CMP_CHURN)):
            if st.get("running"):
                return name
        return None

    @app.route("/api/conn")
    def conn_get():
        return jsonify({"ok": True, **conn_public()})

    @app.route("/api/conn/test", methods=["POST"])
    def conn_test():
        body = request.get_json(silent=True) or {}
        try:
            cfg, sch = candidate(body)
            return jsonify({"ok": True, **probe(cfg, sch)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e).strip()}), 400

    @app.route("/api/conn/apply", methods=["POST"])
    def conn_apply():
        body = request.get_json(silent=True) or {}
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running against the "
                            "current connection — wait for it before switching."}), 400
        try:
            cfg, sch = candidate(body)
            info = probe(cfg, sch)          # refuse to switch to something unreachable
        except Exception as e:
            return jsonify({"ok": False, "error": str(e).strip()}), 400

        G["DB_CONFIG"].update(cfg)           # in place: every helper reads this dict
        G["SCHEMA"] = sch
        G["BUILD_HISTORY"].clear()
        CONN_META["source"] = "set from the UI"

        saved = False
        if body.get("save"):
            payload = {"host": cfg["host"], "port": cfg["port"], "user": cfg["user"],
                       "dbname": cfg["database"], "schema": sch}
            if body.get("save_password"):
                payload["password"] = cfg["password"]
            try:
                fd = os.open(CONN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as fh:
                    json.dump(payload, fh, indent=2)
                saved = True
                CONN_META["source"] = f"saved file {CONN_FILE}"
            except Exception as e:
                info["save_error"] = str(e)
        return jsonify({"ok": True, "saved": saved, **info, **conn_public()})

    @app.route("/api/conn/forget", methods=["POST"])
    def conn_forget():
        try:
            os.remove(CONN_FILE)
        except FileNotFoundError:
            pass
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, **conn_public()})

    # ════════════════════════════════════════════════════════════════════════
    #  VECTORCHORD — status, extension, table clone
    # ════════════════════════════════════════════════════════════════════════

    def am_indexes(table, am):
        """Indexes of one access method on one table, with the options parsed out of
        pg_get_indexdef so the UI can show what each was built with."""
        try:
            rows = q("""
                SELECT i.relname AS name, pg_relation_size(i.oid) AS bytes,
                       pg_get_indexdef(i.oid) AS indexdef, x.indisvalid AS valid
                FROM pg_class i
                JOIN pg_index x ON x.indexrelid = i.oid
                JOIN pg_class t ON t.oid = x.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_am am ON am.oid = i.relam
                WHERE n.nspname = %s AND t.relname = %s AND am.amname = %s
                ORDER BY i.relname""", (schema(), table, am))
        except Exception:
            return []
        for r in rows:
            d = r.get("indexdef") or ""
            r["size"] = human_bytes(r["bytes"])
            r["lists"] = None
            m = re.search(r"lists\s*=\s*\[([0-9,\s]*)\]", d)           # vchordrq (TOML)
            if m:
                r["lists"] = [int(x) for x in m.group(1).split(",") if x.strip()]
            else:
                m = re.search(r"lists\s*=\s*'?(\d+)", d)               # ivfflat
                if m:
                    r["lists"] = [int(m.group(1))]
            m = re.search(r"\bm\s*=\s*'?(\d+)", d)
            r["m"] = int(m.group(1)) if m and am == "hnsw" else None
            m = re.search(r"ef_construction\s*=\s*'?(\d+)", d)
            r["ef_construction"] = int(m.group(1)) if m else None
            m = re.search(r"rerank_in_table\s*=\s*(true|false)", d)
            r["rerank_in_table"] = (m.group(1) == "true") if m else False
            m = re.search(r"residual_quantization\s*=\s*(true|false)", d)
            r["residual_quantization"] = (m.group(1) == "true") if m else None
            m = re.search(r"\b((?:vector|halfvec)_\w+_ops)\b", d)
            r["opclass"] = m.group(1) if m else None
        return rows

    def vc_indexes(table=None):
        return am_indexes(table or VC_TABLE, "vchordrq")

    @app.route("/api/vc/status")
    def vc_status():
        out = {"db_ok": False, "vc_table": VC_TABLE, "vp_table": vp_table(),
               "schema": schema(), "clone": CLONE, "build": {k: v for k, v in VC_BUILD.items()
                                                              if k != "notices"}}
        try:
            get_conn().close()
            out["db_ok"] = True
        except Exception as e:
            out["error"] = str(e)
            return jsonify(out)

        out["server_version"] = scalar("SELECT current_setting('server_version')")
        out["extension"] = scalar("SELECT extversion FROM pg_extension WHERE extname='vchord'")
        out["pgvector"] = scalar("SELECT extversion FROM pg_extension WHERE extname='vector'")
        out["preloaded"] = "vchord" in (scalar("SELECT current_setting('shared_preload_libraries')") or "")
        out["maintenance_work_mem"] = scalar("SELECT current_setting('maintenance_work_mem')")
        out["max_parallel_maintenance_workers"] = scalar(
            "SELECT current_setting('max_parallel_maintenance_workers')")
        try:
            out["gucs"] = q("SELECT name, setting FROM pg_settings "
                            "WHERE name LIKE 'vchordrq.%' ORDER BY name")
        except Exception:
            out["gucs"] = []

        vp, vc = vp_table(), VC_TABLE
        out["vp_exists"], out["vc_exists"] = exists(vp), exists(vc)
        if out["vp_exists"]:
            r = q1(f"SELECT count(*) AS n, max(id) AS mx FROM {qual(vp)}")
            out["vp_rows"], out["vp_max_id"], out["dim"] = int(r["n"]), r["mx"], dim_of(vp)
        if out["vc_exists"]:
            r = q1(f"SELECT count(*) AS n, max(id) AS mx FROM {qual(vc)}")
            out["vc_rows"], out["vc_max_id"], out["vc_dim"] = int(r["n"]), r["mx"], dim_of(vc)
            out["vc_size"] = scalar("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                    (f"{schema()}.{vc}",))
            out["indexes"] = vc_indexes()
        if out["vp_exists"] and out["vc_exists"]:
            out["in_sync"] = (out["vp_rows"] == out["vc_rows"]
                              and out["vp_max_id"] == out["vc_max_id"])
        out["advisor"] = vc_advice(out.get("vc_rows") or out.get("vp_rows") or 0)
        return jsonify(out)

    @app.route("/api/vc/setup/extension", methods=["POST"])
    def vc_setup_extension():
        conn = None
        try:
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
            log_query("ddl", "CREATE EXTENSION IF NOT EXISTS vchord CASCADE")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vchord'")
            v = cur.fetchone()
            return jsonify({"ok": True, "version": v[0] if v else None,
                            "notices": clean_notices(conn.notices)})
        except Exception as e:
            msg = str(e)
            if "vchord" in msg and "shared_preload_libraries" in msg:
                msg += ("\n\nvchord must be listed in shared_preload_libraries and the "
                        "server restarted before the extension can be created.")
            return jsonify({"ok": False, "error": msg}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    def vc_advice(rows):
        """Starting points, not rules — the reference's own guidance is a range."""
        if rows < 128_000:
            lists, rule = [], "under ~128k rows an unpartitioned index (lists = []) is the usual start"
            probes = ""
        else:
            n = int(round(4 * math.sqrt(rows)))
            lists, rule = [n], "single level, about 4 × sqrt(rows)"
            probes = str(max(1, int(round(0.05 * n))))
        return {"lists": lists, "rule": rule, "probes_start": probes,
                "note": "probes: roughly 3–10% of lists. For lists = [] set probes to ''. "
                        "Re-measure recall after changing either."}

    @app.route("/api/vc/index/advisor")
    def vc_advisor():
        rows = int(request.args.get("rows") or count_rows(vp_table()) or 0)
        return jsonify({"ok": True, "rows": rows, **vc_advice(rows)})

    def table_ddl(table, dim):
        # Same shape as the Vector Plus table, including the stored tsvector, so
        # table sizes and insert costs are comparable like for like.
        return f"""
            CREATE TABLE {qual(table)} (
                id          bigserial PRIMARY KEY,
                topic       text     NOT NULL,
                category_id int      NOT NULL,
                title       text     NOT NULL,
                body        text     NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now(),
                tsv         tsvector GENERATED ALWAYS AS
                            (to_tsvector('english', title || ' ' || body)) STORED,
                embedding   vector({dim})
            )"""

    def clone_target(table):
        return VC_TARGET if table == VC_TABLE else PGV_TARGET

    def _clone_worker(tables, step=50_000):
        CLONE.update({"running": True, "phase": "preparing", "done": 0, "total": 0,
                      "started": time.time(), "elapsed": 0, "error": None, "result": None,
                      "tables": list(tables), "current": None})
        conn = None
        try:
            vp = vp_table()
            dim = dim_of(vp)
            if not dim:
                raise RuntimeError(f"{vp} does not exist or has no embedding column — "
                                   "create and load it first.")
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.execute(f"SELECT count(*), coalesce(min(id),0), coalesce(max(id),0) FROM {qual(vp)}")
            total, lo, hi = cur.fetchone()
            CLONE["total"] = int(total) * len(tables)
            per = []
            for tb in tables:
                CLONE["current"], CLONE["phase"] = tb, "preparing"
                cur.execute(f"DROP TABLE IF EXISTS {qual(tb)} CASCADE")
                log_query("ddl", f"DROP TABLE IF EXISTS {qual(tb)} CASCADE")
                cur.execute(table_ddl(tb, dim))
                CLONE["phase"] = "copying"
                sql = (f"INSERT INTO {qual(tb)} (id, topic, category_id, title, body, created_at, embedding)\n"
                       f"SELECT id, topic, category_id, title, body, created_at, embedding\n"
                       f"FROM {qual(vp)} WHERE id >= %s AND id < %s")
                t0, copied, at = time.perf_counter(), 0, lo
                while at <= hi:
                    cur.execute(sql, (at, at + step))
                    copied += cur.rowcount
                    CLONE["done"] += cur.rowcount
                    CLONE["elapsed"] = round(time.time() - CLONE["started"], 2)
                    at += step
                copy_s = round(time.perf_counter() - t0, 3)
                log_query("clone", sql, ms=copy_s * 1000, rows=copied, label=tb)
                cur.execute("SELECT setval(pg_get_serial_sequence(%s, 'id'), %s, true)",
                            (f"{schema()}.{tb}", max(1, int(hi))))
                CLONE["phase"] = "indexing"
                t1 = time.perf_counter()
                cur.execute(f'CREATE INDEX "{tb}_cat_idx" ON {qual(tb)} (category_id)')
                cat_s = round(time.perf_counter() - t1, 3)
                CLONE["phase"] = "analyzing"
                cur.execute(f"ANALYZE {qual(tb)}")
                rps = int(copied / copy_s) if copy_s else 0
                per.append({"table": tb, "rows": copied, "copy_seconds": copy_s,
                            "rows_per_sec": rps, "category_index_seconds": cat_s})
                store_insert("loads", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             target=clone_target(tb), rows=copied, dim=dim, seed=None,
                             seconds=copy_s, rows_per_sec=rps, generator=f"clone of {vp}")
            secs = round(sum(p["copy_seconds"] for p in per), 3)
            rows = sum(p["rows"] for p in per)
            CLONE["result"] = {"tables": per, "rows": rows, "dim": dim, "copy_seconds": secs,
                               "rows_per_sec": int(rows / secs) if secs else 0,
                               "category_index_seconds": round(sum(p["category_index_seconds"] for p in per), 3)}
            CLONE["phase"] = "done"
        except Exception as e:
            CLONE["error"] = str(e)
            CLONE["phase"] = "error"
        finally:
            CLONE["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    def start_clone(tables, ext_needed=None):
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        if not exists(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} does not exist yet. "
                            "Create and load the Vector Plus table first."}), 400
        if not count_rows(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} is empty."}), 400
        if ext_needed and not scalar("SELECT 1 FROM pg_extension WHERE extname = %s",
                                     (ext_needed,), default=0):
            return jsonify({"ok": False, "error": f"{ext_needed} is not installed in this database."}), 400
        threading.Thread(target=_clone_worker, args=(list(tables),), daemon=True).start()
        return jsonify({"ok": True, "tables": list(tables)})

    @app.route("/api/vc/table/clone", methods=["POST"])
    def vc_table_clone():
        return start_clone([VC_TABLE], "vchord")

    @app.route("/api/vc/table/clone/status")
    @app.route("/api/pgv/table/clone/status")
    def vc_clone_status():
        s = dict(CLONE)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 2)
        s["pct"] = round(100 * s["done"] / s["total"], 1) if s["total"] else 0
        return jsonify(s)


    # ════════════════════════════════════════════════════════════════════════
    #  VECTORCHORD — index build
    # ════════════════════════════════════════════════════════════════════════

    def toml_options(lists, rq, rerank_in_table, threads, spherical):
        top = []
        if rq is not None:
            top.append(f"residual_quantization = {'true' if rq else 'false'}")
        if rerank_in_table:
            top.append("rerank_in_table = true")
        inner = [f"lists = [{', '.join(str(x) for x in lists)}]"]
        if threads:
            inner.append(f"build_threads = {int(threads)}")
        if spherical is not None:
            inner.append(f"spherical_centroids = {'true' if spherical else 'false'}")
        return "\n".join(top + ["[build.internal]"] + inner)

    def build_sql(name, opclass, dim, concurrently, options):
        col = "embedding"
        if opclass.startswith("halfvec"):
            col = f"(embedding::halfvec({dim}))"
        return (f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}\"{name}\" "
                f"ON {qual(VC_TABLE)}\n    USING vchordrq ({col} {opclass})\n"
                f"    WITH (options = $${chr(10)}{options}{chr(10)}$$)")

    def _vc_progress(pid):
        while VC_BUILD["running"]:
            try:
                r = q1("""SELECT phase, blocks_done, blocks_total, tuples_done, tuples_total
                          FROM pg_stat_progress_create_index WHERE pid = %s""",
                       (pid,), timeout_ms=3000)
                if r:
                    VC_BUILD["progress"] = r
            except Exception:
                pass
            time.sleep(0.5)

    def _vc_build_worker(sql, name, lists, opclass, concurrently, maint_mem, workers, note, to_drop=None):
        VC_BUILD.update({"running": True, "sql": sql, "started": time.time(), "elapsed": 0,
                         "error": None, "notices": [], "phase": "building",
                         "progress": None, "result": None, "name": name,
                         "maintenance_work_mem": maint_mem})
        conn = None
        try:
            rows = count_rows(VC_TABLE)
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SELECT pg_backend_pid()")
            pid = cur.fetchone()[0]
            VC_BUILD["pid"] = pid
            threading.Thread(target=_vc_progress, args=(pid,), daemon=True).start()
            cur.execute("SET statement_timeout = 0")
            for old in to_drop or []:
                cur.execute(f'DROP INDEX IF EXISTS "{schema()}"."{old}"')
                log_query("ddl", f'DROP INDEX IF EXISTS "{schema()}"."{old}"', label=old)
            if maint_mem:
                cur.execute("SET maintenance_work_mem = %s", (maint_mem,))
            if workers is not None:
                cur.execute("SET max_parallel_maintenance_workers = %s", (int(workers),))
            t0 = time.perf_counter()
            cur.execute(sql)
            elapsed = time.perf_counter() - t0
            log_query("index-build-vchord", sql, ms=elapsed * 1000, rows=rows, label=name,
                      settings=[f"maintenance_work_mem = {maint_mem}"] if maint_mem else None)
            VC_BUILD["notices"] = clean_notices(conn.notices)
            VC_BUILD["elapsed"] = round(elapsed, 3)
            size_b = scalar("SELECT pg_relation_size(%s::regclass)",
                            (f"{schema()}.{name}",), default=0) or 0
            rec = {"name": name, "lists": lists, "opclass": opclass,
                   "seconds": round(elapsed, 3), "rows": rows,
                   "rows_per_sec": int(rows / elapsed) if elapsed > 0 else 0,
                   "bytes": int(size_b), "size": human_bytes(size_b),
                   "bytes_per_row": round(size_b / rows, 1) if rows else 0,
                   "maintenance_work_mem": maint_mem, "notices": VC_BUILD["notices"],
                   "sql": sql, "target": VC_TARGET}
            rec["id"] = store_insert(
                "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                target=VC_TARGET, label=VC_LABEL, engine="vchordrq", am="vchordrq",
                dsn_db=G["DB_CONFIG"].get("database"), rows=rows, dim=dim_of(VC_TABLE),
                lists=(lists[0] if lists else 0), opclass=opclass,
                concurrently=int(bool(concurrently)), index_name=name,
                build_seconds=round(elapsed, 3),
                rows_per_sec=round(rows / elapsed, 1) if elapsed > 0 else 0,
                index_bytes=int(size_b),
                bytes_per_row=round(size_b / rows, 2) if rows else 0,
                maint_work_mem=maint_mem,
                parallel_workers=scalar("SELECT current_setting('max_parallel_maintenance_workers')"),
                pg_version=scalar("SELECT current_setting('server_version')"),
                ext_version=scalar("SELECT extversion FROM pg_extension WHERE extname='vchord'"),
                notices=" | ".join(VC_BUILD["notices"]), note=note)
            VC_BUILD["result"] = rec
            VC_BUILD["phase"] = "done"
        except Exception as e:
            VC_BUILD["error"] = str(e)
            VC_BUILD["phase"] = "error"
            if conn:
                VC_BUILD["notices"] = clean_notices(conn.notices)
        finally:
            VC_BUILD["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/vc/index/build/start", methods=["POST"])
    def vc_build_start():
        if VC_BUILD["running"]:
            return jsonify({"ok": False, "error": "A VectorChord build is already running"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        body = request.get_json(silent=True) or {}

        lists_raw = str(body.get("lists", "") if body.get("lists") is not None else "")
        if not _LISTS_RE.match(lists_raw):
            return jsonify({"ok": False, "error": "lists must be empty, one number, or two "
                            "comma-separated numbers — e.g. 1000 or 1000,50000"}), 400
        lists = [int(x) for x in lists_raw.split(",") if x.strip()]
        if any(x < 1 or x > 1_000_000 for x in lists):
            return jsonify({"ok": False, "error": "each lists value must be 1–1,000,000"}), 400

        opclass = body.get("opclass", "vector_l2_ops")
        if opclass not in G["OPCLASSES"]:
            return jsonify({"ok": False, "error": f"Unknown opclass {opclass}"}), 400
        rq  = body.get("residual_quantization")           # None → leave at server default
        rq  = None if rq in (None, "", "default") else bool(rq)
        rit = bool(body.get("rerank_in_table", False))
        sph = body.get("spherical_centroids")
        sph = None if sph in (None, "", "default") else bool(sph)
        threads = body.get("build_threads")
        threads = int(threads) if str(threads or "").strip() else None
        if threads is not None and not (1 <= threads <= 255):
            return jsonify({"ok": False, "error": "build_threads must be 1–255"}), 400
        workers = body.get("parallel_workers")
        workers = int(workers) if str(workers if workers is not None else "").strip() else None
        concurrently = bool(body.get("concurrently", False))
        maint = (body.get("maintenance_work_mem") or "").strip() or None
        if maint and not _MWM_RE.match(maint):
            return jsonify({"ok": False, "error": "maintenance_work_mem must look like 512MB or 2GB"}), 400

        if not exists(VC_TABLE):
            return jsonify({"ok": False, "error": f"{VC_TABLE} does not exist — clone it first."}), 400
        rows = count_rows(VC_TABLE)
        if not rows:
            return jsonify({"ok": False, "error": f"{VC_TABLE} is empty — clone it first."}), 400
        if not scalar("SELECT 1 FROM pg_extension WHERE extname='vchord'", default=0):
            return jsonify({"ok": False, "error": "vchord is not installed in this database."}), 400
        dim = dim_of(VC_TABLE)

        existing_vc = vc_indexes()
        to_drop = [i["name"] for i in existing_vc] if body.get("replace", True) else []

        name = (body.get("name") or "").strip()
        if name:
            if not re.fullmatch(r"[A-Za-z0-9_]{1,60}", name):
                return jsonify({"ok": False, "error": "Index name must be letters, digits and underscores"}), 400
        else:
            short = opclass.replace("vector_", "v").replace("halfvec_", "h").replace("_ops", "")
            tag = "0" if not lists else "_".join(str(x) for x in lists)
            base, taken, n = f"{VC_PREFIX}_vchordrq_l{tag}_{short}", {r["name"] for r in existing_vc if r["name"] not in to_drop}, 2
            name = base
            while name in taken:
                name, n = f"{base}_{n}", n + 1

        options = toml_options(lists, rq, rit, threads, sph)
        sql = build_sql(name, opclass, dim, concurrently, options)

        warnings = []
        if lists and rows < 128_000:
            warnings.append(f"{rows:,} rows is small for a partitioned index; lists = [] "
                            "is the usual start below ~128k rows.")
        if lists and rows / max(1, lists[0]) < 39:
            warnings.append(f"Only {rows / lists[0]:.0f} rows per list — k-means needs many "
                            "more points per centroid than that to be useful.")
        if rq and "l2" not in opclass:
            warnings.append("residual_quantization is intended for L2 distance; the server may reject it here.")
        if rit:
            warnings.append("rerank_in_table = true makes a capacity-optimized index: it is "
                            "smaller, and queries rerank from the heap, so expect higher latency.")
        if not concurrently:
            warnings.append("CREATE INDEX blocks writes on the table. Use CONCURRENTLY in production.")
        pre = "vchord" in (scalar("SELECT current_setting('shared_preload_libraries')") or "")
        if not pre:
            warnings.append("vchord is not in shared_preload_libraries on this server; "
                            "index builds and the vchordrq.* settings need it.")

        if to_drop:
            warnings.append(f"Replacing existing index {', '.join(to_drop)} — it is dropped first and "
                            "not counted in the build time.")
        note = " ".join(options.split())
        threading.Thread(target=_vc_build_worker,
                         args=(sql, name, lists, opclass, concurrently, maint, workers, note, to_drop),
                         daemon=True).start()
        return jsonify({"ok": True, "sql": sql, "name": name, "warnings": warnings,
                        "advisor": vc_advice(rows)})

    @app.route("/api/vc/index/build/status")
    def vc_build_status():
        s = dict(VC_BUILD)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 2)
        return jsonify(s)

    @app.route("/api/vc/index/list")
    def vc_index_list():
        hist = store_query("SELECT * FROM builds WHERE target = ? ORDER BY id DESC LIMIT 50",
                           (VC_TARGET,))
        for h in hist:
            h["at"] = (h.get("ts") or "")[5:16]
            h["size"] = human_bytes(h.get("index_bytes"))
        return jsonify({"ok": True, "rows": vc_indexes(), "history": hist})

    @app.route("/api/vc/index/drop", methods=["POST"])
    def vc_index_drop():
        body = request.get_json(silent=True) or {}
        name = body.get("name") or ""
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            return jsonify({"ok": False, "error": "Bad index name"}), 400
        conn = None
        try:
            conn = get_conn(autocommit=True)
            sql = f'DROP INDEX IF EXISTS "{schema()}"."{name}"'
            conn.cursor().execute(sql)
            log_query("ddl", sql, label=name)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/vc/index/prewarm", methods=["POST"])
    def vc_index_prewarm():
        """vchordrq is disk-first: the first query pays for loading the index."""
        body = request.get_json(silent=True) or {}
        name = body.get("name") or ""
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            return jsonify({"ok": False, "error": "Bad index name"}), 400
        conn = None
        try:
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            sql = "SELECT vchordrq_prewarm(%s::regclass)"
            t0 = time.perf_counter()
            cur.execute(sql, (f"{schema()}.{name}",))
            res = cur.fetchone()[0]
            el = round(time.perf_counter() - t0, 3)
            log_query("prewarm", sql, ms=el * 1000, label=name,
                      sql_full=resolve_sql(cur, sql, (f"{schema()}.{name}",)))
            return jsonify({"ok": True, "seconds": el, "result": res})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    # ════════════════════════════════════════════════════════════════════════
    #  PGVECTOR — HNSW and IVFFlat, each on its own clone of the table
    # ════════════════════════════════════════════════════════════════════════

    def pgv_tables():
        return {"hnsw": PGH_TABLE, "ivfflat": PGI_TABLE}

    def pgv_advice(rows):
        """pgvector's own README guidance, made clickable."""
        if rows <= 1_000_000:
            lists, rule = max(10, rows // 1000), "rows / 1000 up to 1M rows"
        else:
            lists, rule = int(math.sqrt(rows)), "sqrt(rows) above 1M rows"
        return {"ivfflat_lists": lists, "ivfflat_rule": rule,
                "ivfflat_probes_start": max(1, int(math.sqrt(lists))),
                "hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search_start": 40,
                "note": "IVFFlat: probes ≈ sqrt(lists) to start; build it AFTER loading. "
                        "HNSW: no training step, but the graph should fit in maintenance_work_mem "
                        "or the build slows sharply. ef_search must be at least your LIMIT."}

    @app.route("/api/pgv/status")
    def pgv_status():
        out = {"db_ok": False, "schema": schema(), "vp_table": vp_table(),
               "clone": CLONE, "build": {k: v for k, v in PGV_BUILD.items() if k != "notices"}}
        try:
            get_conn().close()
            out["db_ok"] = True
        except Exception as e:
            out["error"] = str(e)
            return jsonify(out)
        out["pgvector"] = scalar("SELECT extversion FROM pg_extension WHERE extname='vector'")
        out["maintenance_work_mem"] = scalar("SELECT current_setting('maintenance_work_mem')")
        out["max_parallel_maintenance_workers"] = scalar(
            "SELECT current_setting('max_parallel_maintenance_workers')")
        try:
            out["gucs"] = q("SELECT name, setting FROM pg_settings WHERE name LIKE 'hnsw.%' "
                            "OR name LIKE 'ivfflat.%' ORDER BY name")
        except Exception:
            out["gucs"] = []
        out["vp_exists"] = exists(vp_table())
        vp_rows = vp_max = None
        if out["vp_exists"]:
            r = q1(f"SELECT count(*) AS n, max(id) AS mx FROM {qual(vp_table())}")
            vp_rows, vp_max = int(r["n"]), r["mx"]
            out["vp_rows"], out["dim"] = vp_rows, dim_of(vp_table())
        out["tables"] = {}
        for am, tb in pgv_tables().items():
            t = {"table": tb, "exists": exists(tb), "am": am}
            if t["exists"]:
                r = q1(f"SELECT count(*) AS n, max(id) AS mx FROM {qual(tb)}")
                t["rows"] = int(r["n"])
                t["size"] = scalar("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                   (f"{schema()}.{tb}",))
                t["indexes"] = am_indexes(tb, am)
                if vp_rows is not None:
                    t["in_sync"] = (t["rows"] == vp_rows and r["mx"] == vp_max)
            out["tables"][am] = t
        out["advisor"] = pgv_advice(vp_rows or 0)
        return jsonify(out)

    @app.route("/api/pgv/table/clone", methods=["POST"])
    def pgv_table_clone():
        which = (request.get_json(silent=True) or {}).get("which", "both")
        if which not in ("hnsw", "ivfflat", "both"):
            return jsonify({"ok": False, "error": "which must be hnsw, ivfflat or both"}), 400
        tbls = list(pgv_tables().values()) if which == "both" else [pgv_tables()[which]]
        return start_clone(tbls, "vector")

    def _pgv_progress(pid):
        while PGV_BUILD["running"]:
            try:
                r = q1("""SELECT phase, blocks_done, blocks_total, tuples_done, tuples_total
                          FROM pg_stat_progress_create_index WHERE pid = %s""",
                       (pid,), timeout_ms=3000)
                if r:
                    PGV_BUILD["progress"] = r
            except Exception:
                pass
            time.sleep(0.5)

    def _pgv_build_worker(sql, name, table, am, opclass, opts, concurrently,
                          maint_mem, workers, to_drop):
        PGV_BUILD.update({"running": True, "sql": sql, "started": time.time(), "elapsed": 0,
                          "error": None, "notices": [], "phase": "building", "progress": None,
                          "result": None, "name": name, "am": am,
                          "maintenance_work_mem": maint_mem})
        conn = None
        try:
            rows = count_rows(table)
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SELECT pg_backend_pid()")
            pid = cur.fetchone()[0]
            PGV_BUILD["pid"] = pid
            threading.Thread(target=_pgv_progress, args=(pid,), daemon=True).start()
            cur.execute("SET statement_timeout = 0")
            for old in to_drop or []:
                cur.execute(f'DROP INDEX IF EXISTS "{schema()}"."{old}"')
                log_query("ddl", f'DROP INDEX IF EXISTS "{schema()}"."{old}"', label=old)
            if maint_mem:
                cur.execute("SET maintenance_work_mem = %s", (maint_mem,))
            if workers is not None:
                cur.execute("SET max_parallel_maintenance_workers = %s", (int(workers),))
            t0 = time.perf_counter()
            cur.execute(sql)
            elapsed = time.perf_counter() - t0
            log_query("index-build-pgvector", sql, ms=elapsed * 1000, rows=rows, label=name,
                      settings=[f"maintenance_work_mem = {maint_mem}"] if maint_mem else None)
            PGV_BUILD["notices"] = clean_notices(conn.notices)
            PGV_BUILD["elapsed"] = round(elapsed, 3)
            size_b = scalar("SELECT pg_relation_size(%s::regclass)",
                            (f"{schema()}.{name}",), default=0) or 0
            spilled = any("no longer fits" in n for n in PGV_BUILD["notices"])
            rec = {"name": name, "am": am, "table": table, "opclass": opclass, "opts": opts,
                   "seconds": round(elapsed, 3), "rows": rows,
                   "rows_per_sec": int(rows / elapsed) if elapsed > 0 else 0,
                   "bytes": int(size_b), "size": human_bytes(size_b),
                   "bytes_per_row": round(size_b / rows, 1) if rows else 0,
                   "maintenance_work_mem": maint_mem, "notices": PGV_BUILD["notices"],
                   "graph_spilled": spilled, "sql": sql, "target": PGV_TARGET}
            rec["id"] = store_insert(
                "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                target=PGV_TARGET, label=("pgvector HNSW" if am == "hnsw" else "pgvector IVFFlat"),
                engine=am, am=am, dsn_db=G["DB_CONFIG"].get("database"), rows=rows,
                dim=dim_of(table), lists=opts.get("lists"), m=opts.get("m"),
                ef_construction=opts.get("ef_construction"), opclass=opclass,
                concurrently=int(bool(concurrently)), index_name=name,
                build_seconds=round(elapsed, 3),
                rows_per_sec=round(rows / elapsed, 1) if elapsed > 0 else 0,
                index_bytes=int(size_b), bytes_per_row=round(size_b / rows, 2) if rows else 0,
                maint_work_mem=maint_mem,
                parallel_workers=scalar("SELECT current_setting('max_parallel_maintenance_workers')"),
                pg_version=scalar("SELECT current_setting('server_version')"),
                ext_version=scalar("SELECT extversion FROM pg_extension WHERE extname='vector'"),
                notices=" | ".join(PGV_BUILD["notices"]),
                note=("graph spilled out of maintenance_work_mem" if spilled else None))
            PGV_BUILD["result"] = rec
            PGV_BUILD["phase"] = "done"
        except Exception as e:
            PGV_BUILD["error"] = str(e)
            PGV_BUILD["phase"] = "error"
            if conn:
                PGV_BUILD["notices"] = clean_notices(conn.notices)
        finally:
            PGV_BUILD["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/pgv/index/build/start", methods=["POST"])
    def pgv_build_start():
        if PGV_BUILD["running"]:
            return jsonify({"ok": False, "error": "A pgvector build is already running"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        body = request.get_json(silent=True) or {}
        am = body.get("am", "hnsw")
        if am not in ("hnsw", "ivfflat"):
            return jsonify({"ok": False, "error": "am must be hnsw or ivfflat"}), 400
        opclass = body.get("opclass", "vector_l2_ops")
        if opclass not in G["OPCLASSES"]:
            return jsonify({"ok": False, "error": f"Unknown opclass {opclass}"}), 400
        table = pgv_tables()[am]
        if not exists(table):
            return jsonify({"ok": False, "error": f"{table} does not exist — clone it first."}), 400
        rows = count_rows(table)
        if not rows:
            return jsonify({"ok": False, "error": f"{table} is empty — clone it first."}), 400
        dim = dim_of(table)
        if not opclass.startswith("halfvec") and dim > 2000:
            return jsonify({"ok": False, "error": f"pgvector indexes vector columns up to 2000 "
                            f"dimensions; this is {dim}. Use a halfvec opclass (up to 4000)."}), 400

        try:
            if am == "hnsw":
                m_ = int(body.get("m") or 16)
                efc = int(body.get("ef_construction") or 64)
                if not (2 <= m_ <= 100):
                    raise ValueError("m must be 2–100")
                if not (4 <= efc <= 1000) or efc < 2 * m_:
                    raise ValueError("ef_construction must be 4–1000 and at least 2 × m")
                opts, with_ = {"m": m_, "ef_construction": efc}, f"m = {m_}, ef_construction = {efc}"
                tag = f"m{m_}_ef{efc}"
            else:
                lists = int(body.get("lists") or max(10, rows // 1000))
                if not (1 <= lists <= 32768):
                    raise ValueError("lists must be 1–32768")
                opts, with_, tag = {"lists": lists}, f"lists = {lists}", f"l{lists}"
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        workers = body.get("parallel_workers")
        workers = int(workers) if str(workers if workers is not None else "").strip() else None
        concurrently = bool(body.get("concurrently", False))
        maint = (body.get("maintenance_work_mem") or "").strip() or None
        if maint and not _MWM_RE.match(maint):
            return jsonify({"ok": False, "error": "maintenance_work_mem must look like 512MB or 2GB"}), 400
        existing = am_indexes(table, am)
        to_drop = [i["name"] for i in existing] if body.get("replace", True) else []

        name = (body.get("name") or "").strip()
        if name:
            if not re.fullmatch(r"[A-Za-z0-9_]{1,60}", name):
                return jsonify({"ok": False, "error": "Index name must be letters, digits and underscores"}), 400
        else:
            short = opclass.replace("vector_", "v").replace("halfvec_", "h").replace("_ops", "")
            base = f"{table}_emb_{am}_{tag}_{short}"
            taken = {i["name"] for i in existing if i["name"] not in to_drop}
            name, n = base, 2
            while name in taken:
                name, n = f"{base}_{n}", n + 1

        col = f"(embedding::halfvec({dim}))" if opclass.startswith("halfvec") else "embedding"
        sql = (f"CREATE INDEX {'CONCURRENTLY ' if concurrently else ''}\"{name}\" ON {qual(table)}\n"
               f"    USING {am} ({col} {opclass})\n    WITH ({with_})")

        warnings = []
        if to_drop:
            warnings.append(f"Replacing existing index {', '.join(to_drop)} — it is dropped first and "
                            "not counted in the build time.")
        if am == "ivfflat" and opts["lists"] > 0 and rows / opts["lists"] < 39:
            warnings.append(f"Only {rows / opts['lists']:.0f} rows per list — IVFFlat k-means needs far "
                            "more points per centroid to be useful.")
        if am == "hnsw" and not maint:
            warnings.append("No maintenance_work_mem given. If the HNSW graph does not fit, the build "
                            "slows sharply and the server logs 'graph no longer fits' — set it higher.")
        if not concurrently:
            warnings.append("CREATE INDEX blocks writes on the table. Use CONCURRENTLY in production.")
        threading.Thread(target=_pgv_build_worker,
                         args=(sql, name, table, am, opclass, opts, concurrently, maint, workers, to_drop),
                         daemon=True).start()
        return jsonify({"ok": True, "sql": sql, "name": name, "warnings": warnings,
                        "advisor": pgv_advice(rows)})

    @app.route("/api/pgv/index/build/status")
    def pgv_build_status():
        s = dict(PGV_BUILD)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 2)
        return jsonify(s)

    @app.route("/api/pgv/index/list")
    def pgv_index_list():
        rows = []
        for am, tb in pgv_tables().items():
            for r in am_indexes(tb, am):
                r["am"], r["table"] = am, tb
                rows.append(r)
        hist = store_query("SELECT * FROM builds WHERE target = ? ORDER BY id DESC LIMIT 50",
                           (PGV_TARGET,))
        for h in hist:
            h["at"] = (h.get("ts") or "")[5:16]
            h["size"] = human_bytes(h.get("index_bytes"))
        return jsonify({"ok": True, "rows": rows, "history": hist})

    @app.route("/api/pgv/index/drop", methods=["POST"])
    def pgv_index_drop():
        return vc_index_drop()

    # ════════════════════════════════════════════════════════════════════════
    #  CACHE CONTROL — so a "cold" run really is cold
    # ════════════════════════════════════════════════════════════════════════
    # Three layers sit between a query and the disk, and each needs its own answer:
    #   1. PostgreSQL shared_buffers  → pg_buffercache_evict_relation() (PG 18),
    #      or pg_buffercache_evict() per buffer (PG 17). Needs the pg_buffercache
    #      extension and superuser or the right grants. Not available before 17.
    #   2. The OS page cache          → posix_fadvise(DONTNEED) on exactly the
    #      relation's files. Needs no root, only read access to the data directory,
    #      so it works when this app runs as the postgres user on the DB host.
    #      Optionally VP_DROP_CACHES_CMD for a full `echo 3 > drop_caches`.
    #   3. The storage's own cache    → not reachable from here; say so.
    # Dirty pages cannot be dropped, so a CHECKPOINT comes first.

    DROP_CMD = os.environ.get("VP_DROP_CACHES_CMD", "").strip()

    def relation_oids(cur, table):
        cur.execute("""
            WITH t AS (SELECT %s::regclass::oid AS oid),
                 base AS (SELECT oid FROM t
                          UNION SELECT c.reltoastrelid FROM pg_class c JOIN t ON c.oid = t.oid
                           WHERE c.reltoastrelid <> 0)
            SELECT oid FROM base
            UNION SELECT indexrelid FROM pg_index WHERE indrelid IN (SELECT oid FROM base)""",
                    (f"{schema()}.{table}",))
        return [r[0] for r in cur.fetchall()]

    def cache_caps(cur):
        caps = {"errors": []}
        cur.execute("SELECT current_setting('is_superuser') = 'on', "
                    "current_setting('server_version_num')::int, current_setting('shared_buffers')")
        caps["superuser"], ver, caps["shared_buffers"] = cur.fetchone()
        caps["pg_major"] = ver // 10000
        try:
            cur.execute("SHOW data_directory")
            caps["data_directory"] = cur.fetchone()[0]
        except Exception:
            caps["data_directory"] = None
        dd = caps["data_directory"]
        caps["same_host"] = bool(dd and os.path.isdir(dd) and os.access(dd, os.R_OK))
        caps["os_evict"] = bool(caps["same_host"] and hasattr(os, "posix_fadvise"))
        cur.execute("SELECT extname, extversion FROM pg_extension "
                    "WHERE extname IN ('pg_buffercache', 'pg_prewarm')")
        caps["extensions"] = {n: v for n, v in cur.fetchall()}
        cur.execute("SELECT proname FROM pg_proc WHERE proname IN "
                    "('pg_buffercache_evict_relation', 'pg_buffercache_evict')")
        have = {r[0] for r in cur.fetchall()}
        caps["shared_evict"] = ("relation" if "pg_buffercache_evict_relation" in have
                                else "buffer" if "pg_buffercache_evict" in have else None)
        caps["drop_caches_cmd"] = bool(DROP_CMD)
        try:
            cur.execute("SELECT pg_has_role(current_user, 'pg_checkpoint', 'member')")
            caps["can_checkpoint"] = bool(caps["superuser"] or cur.fetchone()[0])
        except Exception:
            caps["can_checkpoint"] = bool(caps["superuser"])
        caps["can_clean"] = bool(caps["os_evict"] or caps["shared_evict"] or caps["drop_caches_cmd"])
        return caps

    def clean_caches(cur, table):
        """Evict one table, its indexes and its TOAST from every cache layer reachable."""
        t0 = time.perf_counter()
        out = {"table": table, "shared": None, "os": None, "cmd": None, "errors": []}
        caps = cache_caps(cur)
        oids = relation_oids(cur, table)
        try:
            cur.execute("CHECKPOINT")
        except Exception as e:
            out["errors"].append("CHECKPOINT: " + str(e).strip().splitlines()[0])

        if caps["shared_evict"]:
            ev = fl = sk = 0
            for oid in oids:
                try:
                    if caps["shared_evict"] == "relation":
                        cur.execute("SELECT * FROM pg_buffercache_evict_relation(%s::oid::regclass)", (oid,))
                        a, b, c = cur.fetchone()
                        ev, fl, sk = ev + a, fl + b, sk + c
                    else:
                        cur.execute("""SELECT count(*) FILTER (WHERE pg_buffercache_evict(bufferid)) ,
                                              count(*)
                                       FROM pg_buffercache
                                       WHERE reldatabase = (SELECT oid FROM pg_database
                                                            WHERE datname = current_database())
                                         AND relfilenode = pg_relation_filenode(%s::oid::regclass)""", (oid,))
                        a, n = cur.fetchone()
                        ev, sk = ev + a, sk + (n - a)
                except Exception as e:
                    out["errors"].append("shared_buffers: " + str(e).strip().splitlines()[0])
                    break
            out["shared"] = {"evicted": ev, "flushed": fl, "skipped": sk}

        if caps["os_evict"]:
            files = nbytes = 0
            try:
                for oid in oids:
                    cur.execute("SELECT pg_relation_filepath(%s::oid::regclass)", (oid,))
                    rel = cur.fetchone()[0]
                    for seg in range(0, 4096):                    # 1 GB segments
                        path = os.path.join(caps["data_directory"], rel + ("" if seg == 0 else f".{seg}"))
                        if not os.path.exists(path):
                            break
                        fd = os.open(path, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            nbytes += os.fstat(fd).st_size
                            files += 1
                        finally:
                            os.close(fd)
                out["os"] = {"files": files, "bytes": nbytes}
            except Exception as e:
                out["errors"].append("OS page cache: " + str(e))
        if DROP_CMD:
            try:
                import shlex, subprocess
                r = subprocess.run(shlex.split(DROP_CMD), capture_output=True, text=True, timeout=60)
                out["cmd"] = {"returncode": r.returncode, "stderr": r.stderr.strip()[:200]}
                if r.returncode:
                    out["errors"].append(f"VP_DROP_CACHES_CMD exited {r.returncode}: {r.stderr.strip()[:120]}")
            except Exception as e:
                out["errors"].append("VP_DROP_CACHES_CMD: " + str(e))
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def require_cold(caps):
        if not caps["can_clean"]:
            raise ValueError(
                "Cold runs need a way to clear caches, and none is available here. "
                "Either run this app on the database host as the postgres user (then the OS page "
                "cache can be dropped per file), and/or enable pg_buffercache on PostgreSQL 17+ "
                "(Engine Compare → Cache tools → Enable), and/or set VP_DROP_CACHES_CMD.")

    def prewarm(cur, e):
        """Best effort: pull the index and heap into cache so a warm run starts warm."""
        done, caps_ext = [], {}
        try:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_prewarm'")
            caps_ext = bool(cur.fetchone())
        except Exception:
            caps_ext = False
        for i in engine_indexes(e):
            try:
                if e["key"] == "vchordrq":
                    cur.execute("SELECT vchordrq_prewarm(%s::regclass)", (f"{schema()}.{i['name']}",))
                    done.append("vchordrq_prewarm")
                elif caps_ext:
                    cur.execute("SELECT pg_prewarm(%s::regclass)", (f"{schema()}.{i['name']}",))
                    done.append("pg_prewarm(index)")
            except Exception as ex:
                done.append("index: " + str(ex).strip().splitlines()[0])
        if caps_ext:
            try:
                cur.execute("SELECT pg_prewarm(%s::regclass)", (f"{schema()}.{e['table']}",))
                done.append("pg_prewarm(heap)")
            except Exception as ex:
                done.append("heap: " + str(ex).strip().splitlines()[0])
        return done or ["none — settled by a throwaway pass per setting"]

    def explain_buffers(cur, sql, params):
        cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
        raw = cur.fetchone()[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        plan = raw[0]["Plan"]
        return {"ms": round(raw[0].get("Execution Time", 0), 3),
                "hit": plan.get("Shared Hit Blocks", 0), "read": plan.get("Shared Read Blocks", 0)}

    @app.route("/api/cache/status")
    def cache_status():
        conn = None
        try:
            conn = get_conn(autocommit=True)
            return jsonify({"ok": True, **cache_caps(conn.cursor())})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/cache/setup", methods=["POST"])
    def cache_setup():
        """CREATE (or update) the contrib extensions the cache tools use."""
        conn = None
        res = {}
        try:
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            for ext in ("pg_buffercache", "pg_prewarm"):
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                    cur.execute(f"ALTER EXTENSION {ext} UPDATE")
                    res[ext] = "ready"
                except Exception as e:
                    res[ext] = str(e).strip().splitlines()[0]
            return jsonify({"ok": True, "extensions": res, **cache_caps(cur)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/cache/test", methods=["POST"])
    def cache_test():
        """Prove the cleaning works: per engine, one query cold then warm, with the
        number of shared-buffer blocks it had to read from outside shared_buffers."""
        body = request.get_json(silent=True) or {}
        conn = None
        try:
            es = [e for e in engines() if exists(e["table"]) and engine_indexes(e)
                  and (not body.get("engines") or e["key"] in body["engines"])]
            if not es:
                raise ValueError("No engine has both a table and an index yet")
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.execute("SET jit = off")
            caps = cache_caps(cur)
            require_cold(caps)
            opclass = body.get("opclass", "vector_l2_ops")
            out = []
            for e in es:
                qv = sample_queries(cur, e["table"], 1)[0]
                lit = vec_literal(qv)
                sql = ann_sql(e["table"], opclass, 10, None)
                lists = (engine_indexes(e)[0].get("lists") or [])
                p = {"ivfplus": 16, "vchordrq": default_vc_probes(lists)[min(2, len(default_vc_probes(lists)) - 1)],
                     "hnsw": 40, "ivfflat": default_ivf_probes(lists)[min(2, len(default_ivf_probes(lists)) - 1)]}[e["key"]]
                set_scan_mode(cur, False, True)
                for s in engine_setters(e["key"], p):
                    cur.execute(s)
                cl = clean_caches(cur, e["table"])
                cold = explain_buffers(cur, sql, (lit, lit))
                warm = explain_buffers(cur, sql, (lit, lit))
                out.append({"key": e["key"], "label": e["label"], "table": e["table"], "clean": cl,
                            "cold": cold, "warm": warm,
                            "effective": cold["read"] > warm["read"] or cold["ms"] > 1.5 * warm["ms"]})
            return jsonify({"ok": True, "caps": caps, "engines": out})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    # ════════════════════════════════════════════════════════════════════════
    #  COMPARISON — shared query machinery
    # ════════════════════════════════════════════════════════════════════════

    def probes_ok(s):
        s = str(s if s is not None else "").replace(" ", "")
        if not _PROBES_RE.match(s):
            raise ValueError("vchordrq.probes must be digits, comma-separated per level — "
                             "e.g. 10 or 20,200, or empty for lists = []")
        return s

    def check_vc_probes(lists, p):
        """vchordrq wants one probes value per level of its lists array, and says
        only 'need N probes, but M provided'. Say what to type instead."""
        have = len([x for x in str(p if p is not None else "").split(",") if x.strip() != ""])
        want = len(lists or [])
        if have != want:
            shape = ("an empty string (the index was built with lists = [])" if want == 0 else
                     f"{want} comma-separated value{'s' if want > 1 else ''}, one per level")
            raise ValueError(f"The vchordrq index has lists = {list(lists or [])}, so vchordrq.probes "
                             f"must be {shape}; you gave '{p}'.")

    def engine_setters(key, p, epsilon=None, prefilter=None, iterative=None):
        """SET statements for one engine. Every value is validated before it is
        inlined, so these are safe to execute as plain text."""
        if key == "ivfplus":
            return [f"SET ivfplus.probes = {int(p)}"]
        if key == "vchordrq":
            out = [f"SET vchordrq.probes = '{probes_ok(p)}'"]
            if epsilon not in (None, ""):
                out.append(f"SET vchordrq.epsilon = {float(epsilon)}")
            if prefilter is not None:
                out.append(f"SET vchordrq.prefilter = {'on' if prefilter else 'off'}")
            return out
        if iterative not in (None, "", "off", "strict_order", "relaxed_order"):
            raise ValueError("iterative_scan must be off, strict_order or relaxed_order")
        if key == "hnsw":
            out = [f"SET hnsw.ef_search = {max(1, min(1000, int(p)))}"]
            if iterative and iterative != "off":
                out.append(f"SET hnsw.iterative_scan = {iterative}")
            return out
        if key == "ivfflat":
            out = [f"SET ivfflat.probes = {max(1, int(p))}"]
            if iterative and iterative != "off":
                out.append(f"SET ivfflat.iterative_scan = {iterative}")
            return out
        raise ValueError(f"unknown engine {key}")

    def ann_sql(table, opclass, limit, category_id):
        meta = G["OPCLASSES"][opclass]
        op, col, cast = meta["op"], "embedding", ""
        if opclass.startswith("halfvec"):
            d = dim_of(table)
            col, cast = f"embedding::halfvec({d})", f"::halfvec({d})"
        where = f"WHERE category_id = {int(category_id)}\n" if category_id not in (None, "", "any") else ""
        return (f"SELECT id, topic, category_id, title,\n"
                f"       {col} {op} %s{cast} AS distance\n"
                f"FROM {qual(table)}\n{where}"
                f"ORDER BY {col} {op} %s{cast}\nLIMIT {int(limit)}")

    def plan_of(cur, sql, params):
        """Which index, if any, the planner actually chose — recall of 1.0 at high
        latency is the signature of a quiet sequential scan."""
        cur.execute("EXPLAIN (FORMAT JSON) " + sql, params)
        raw = cur.fetchone()[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        found = {"nodes": [], "index": None}

        def walk(n):
            found["nodes"].append(n.get("Node Type"))
            if n.get("Index Name") and not found["index"]:
                found["index"] = n["Index Name"]
            for c in n.get("Plans", []) or []:
                walk(c)
        walk(raw[0]["Plan"])
        return found

    def set_scan_mode(cur, exact, force_index):
        # vchordrq insists on a valid probes value even while *planning*, so an exact
        # baseline on a table that carries a vchordrq index fails unless its scan is
        # switched off. Harmless (a placeholder setting) if vchord is not loaded.
        cur.execute(f"SET vchordrq.enable_scan = {'off' if exact else 'on'}")
        if exact:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
            cur.execute("SET enable_seqscan = on")
        else:
            cur.execute("SET enable_indexscan = on")
            cur.execute("SET enable_bitmapscan = on")
            cur.execute(f"SET enable_seqscan = {'off' if force_index else 'on'}")

    def run_ann(cur, table, qv, opclass, limit, category_id, setters,
                exact=False, force_index=True, reps=1, warm=True, kind="cmp", clean=None):
        """One query, run `reps` times. In cold mode (`clean` given) the caches are
        evicted right before the first run, so that run is the cold figure and the
        rest are the warm ones; otherwise a throwaway run precedes the timed ones."""
        sql = ann_sql(table, opclass, limit, category_id)
        lit = vec_literal(qv)
        params = (lit, lit)
        set_scan_mode(cur, exact, force_index)
        if not exact:
            for s in setters:
                cur.execute(s)
        plan = plan_of(cur, sql, params)
        cleaned = None
        if clean and not exact:
            cleaned = clean()
        elif warm and not exact:
            cur.execute(sql, params)
            cur.fetchall()
        times, rows = [], []
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            cur.execute(sql, params)
            rows = cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
        full = resolve_sql(cur, sql, params)
        cold = cleaned is not None
        ms = times[0] if cold else float(np.median(times))
        ms_warm = float(np.median(times[1:])) if cold and len(times) > 1 else None
        log_query(kind, sql, settings=[s.replace("SET ", "") for s in setters] if not exact
                  else ["enable_indexscan = off"], ms=ms, rows=len(rows), sql_full=full,
                  label=table + (" (cold)" if cold else ""))
        return {"rows": [{"id": r[0], "topic": r[1], "category_id": r[2], "title": r[3],
                          "distance": round(float(r[4]), 6)} for r in rows],
                "ms": round(ms, 3), "ms_warm": round(ms_warm, 3) if ms_warm is not None else None,
                "cold": cold, "clean": cleaned, "ms_all": [round(t, 3) for t in times],
                "plan": plan, "sql": sql, "table": table}

    # ── the engine registry ─────────────────────────────────────────────────
    def engines():
        """Every engine this dashboard can compare, each on its own table of
        identical rows. A table that does not exist simply drops out."""
        return [
            {"key": "ivfplus", "table": vp_table(), "am": "ivfplus", "target": G["TARGET_NAME"],
             "label": G["TARGET_LABEL"], "param": "ivfplus.probes"},
            {"key": "vchordrq", "table": VC_TABLE, "am": "vchordrq", "target": VC_TARGET,
             "label": VC_LABEL, "param": "vchordrq.probes"},
            {"key": "hnsw", "table": PGH_TABLE, "am": "hnsw", "target": PGV_TARGET,
             "label": "pgvector HNSW", "param": "hnsw.ef_search"},
            {"key": "ivfflat", "table": PGI_TABLE, "am": "ivfflat", "target": PGV_TARGET,
             "label": "pgvector IVFFlat", "param": "ivfflat.probes"},
        ]

    def engine_indexes(e):
        if e["key"] == "ivfplus":
            return [dict(r, size=human_bytes(r["bytes"]))
                    for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"]
        return am_indexes(e["table"], e["am"])

    def need_engines(keys=None, need_index=True):
        """The engines to compare, validated: at least two, identical row sets, and
        exactly one index each so the planner's choice is never in question."""
        es = [e for e in engines() if exists(e["table"]) and (not keys or e["key"] in keys)]
        if keys:
            miss = [k for k in keys if k not in {e["key"] for e in es}]
            if miss:
                raise ValueError("No table yet for: " + ", ".join(miss) +
                                 " — clone it from its tab first")
        if len(es) < 2:
            raise ValueError("Need at least two engine tables to compare — clone "
                             "vc_docs / pgh_docs / pgi_docs from the VectorChord and pgvector tabs")
        base = es[0]
        d0, n0 = dim_of(base["table"]), count_rows(base["table"])
        for e in es[1:]:
            if dim_of(e["table"]) != d0:
                raise ValueError(f"{e['table']} is {dim_of(e['table'])}-dimensional but "
                                 f"{base['table']} is {d0} — re-clone")
            n = count_rows(e["table"])
            if n != n0:
                raise ValueError(f"{e['table']} has {n:,} rows but {base['table']} has {n0:,} — "
                                 "re-clone so every table holds the same rows")
        if need_index:
            for e in es:
                idx = engine_indexes(e)
                if not idx:
                    raise ValueError(f"{e['table']} has no {e['am']} index — build one first "
                                     f"(or untick {e['label']})")
                if len(idx) > 1 and e["key"] != "ivfplus":
                    raise ValueError(f"{e['table']} has {len(idx)} {e['am']} indexes "
                                     f"({', '.join(i['name'] for i in idx)}). The planner chooses between "
                                     "them, which makes the comparison ambiguous — drop all but one.")
        return es

    def truth_engine(es):
        """Exact scans are identical on every table; run them on one that is not vchordrq's."""
        return next((e for e in es if e["key"] != "vchordrq"), es[0])

    def vp_index_note():
        idx = [r for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"]
        if len(idx) > 1:
            return (f"{vp_table()} has {len(idx)} ivfplus indexes; the planner chose between them "
                    f"({', '.join(i['name'] for i in idx)}). Drop the others for a clean comparison.")
        return None

    def _pmap(body):
        return {"ivfplus": body.get("vp_probes", 10), "vchordrq": body.get("vc_probes", ""),
                "hnsw": body.get("hnsw_ef", 40), "ivfflat": body.get("ivf_probes", 10)}

    # ── summary ─────────────────────────────────────────────────────────────
    @app.route("/api/cmp/summary")
    def cmp_summary():
        out = {"ok": True, "engines": {}, "order": []}
        for e in engines():
            card = {"table": e["table"], "label": e["label"], "engine": e["am"],
                    "key": e["key"], "exists": exists(e["table"])}
            if card["exists"]:
                card["rows"] = count_rows(e["table"])
                card["dim"] = dim_of(e["table"])
                card["heap_bytes"] = int(scalar("SELECT pg_table_size(%s::regclass)",
                                                (f"{schema()}.{e['table']}",), default=0) or 0)
                idx = engine_indexes(e)
                for r in idx:
                    r["size"] = human_bytes(r["bytes"])
                card["indexes"] = idx
                last = store_query("SELECT * FROM builds WHERE target = ? AND engine = ? "
                                   "ORDER BY id DESC LIMIT 1", (e["target"], e["am"]))
                if last:
                    b = last[0]
                    card["last_build"] = {k: b.get(k) for k in
                        ("index_name", "build_seconds", "rows_per_sec", "index_bytes",
                         "bytes_per_row", "lists", "m", "ef_construction", "opclass", "rows", "ts")}
                    card["last_build"]["size"] = human_bytes(b.get("index_bytes"))
            out["engines"][e["key"]] = card
            out["order"].append(e["key"])
        # ratios against the first engine that has a recorded build
        base = next((k for k in out["order"] if out["engines"][k].get("last_build")
                     and out["engines"][k]["last_build"].get("build_seconds")), None)
        out["ratios"], out["baseline"] = {}, base
        if base:
            bb = out["engines"][base]["last_build"]
            for k in out["order"]:
                lb = out["engines"][k].get("last_build")
                if k == base or not lb:
                    continue
                r = {}
                if lb.get("build_seconds") and bb.get("build_seconds"):
                    r["build"] = round(lb["build_seconds"] / bb["build_seconds"], 2)
                if lb.get("index_bytes") and bb.get("index_bytes"):
                    r["size"] = round(lb["index_bytes"] / bb["index_bytes"], 2)
                out["ratios"][k] = r
        return jsonify(out)

    # ── one query, every engine ─────────────────────────────────────────────
    @app.route("/api/cmp/search", methods=["POST"])
    def cmp_search():
        body = request.get_json(silent=True) or {}
        conn = None
        try:
            es = need_engines(body.get("engines"))
            qv, label = G["resolve_query_vector"](body)
            opclass = body.get("opclass", "vector_l2_ops")
            if opclass not in G["OPCLASSES"]:
                raise ValueError("Unknown opclass")
            k = max(1, min(200, int(body.get("limit", 10))))
            cat = body.get("category_id")
            reps = max(1, min(20, int(body.get("reps", 3))))
            force = bool(body.get("force_index", True))
            pm = _pmap(body)
            pre = body.get("prefilter")
            pre = None if pre is None else bool(pre)
            it = body.get("iterative") or "off"

            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 120000")
            cur.execute("SET jit = off")
            cold = body.get("cache_mode") == "cold"
            caps = cache_caps(cur)
            if cold:
                require_cold(caps)
                reps = max(reps, 3)           # first run is cold, the rest give the warm figure
            exact = run_ann(cur, truth_engine(es)["table"], qv, opclass, k, cat, [], exact=True, kind="cmp-exact")
            truth = {r["id"] for r in exact["rows"]}

            outs = []
            for e in es:
                if e["key"] == "vchordrq":
                    check_vc_probes(engine_indexes(e)[0].get("lists"), pm["vchordrq"])
                setters = engine_setters(e["key"], pm[e["key"]], body.get("epsilon"), pre, it)
                res = run_ann(cur, e["table"], qv, opclass, k, cat, setters,
                              force_index=force, reps=reps, kind="cmp-" + e["key"],
                              clean=(lambda t=e["table"]: clean_caches(cur, t)) if cold else None)
                ids = {r["id"] for r in res["rows"]}
                outs.append({"key": e["key"], "label": e["label"], "table": e["table"],
                             "result": res, "settings": setters, "ids": sorted(ids),
                             "recall": round(len(truth & ids) / len(truth), 4) if truth else None,
                             "returned": len(res["rows"]),
                             "speedup": round(exact["ms"] / res["ms"], 1) if res["ms"] > 0 else None})
            common = set(outs[0]["ids"])
            for o in outs[1:]:
                common &= set(o["ids"])

            notes = [n for n in (vp_index_note(),) if n]
            vec_names = {i["name"] for e in es for i in engine_indexes(e)}
            for o in outs:
                plan = o["result"]["plan"]
                if plan["index"] and plan["index"] not in vec_names:
                    notes.append(f"{o['label']}: the planner used {plan['index']}, not a vector index, so "
                                 "this result is exact for the filter and says nothing about ANN recall. "
                                 "Selective filters often go this way.")
                if not plan["index"]:
                    notes.append(f"{o['label']}: the planner did not use an index "
                                 f"({', '.join(plan['nodes'])}). Recall will be 1.0 and the latency is a "
                                 f"sequential scan — build an index with opclass {opclass}.")
                if o["key"] == "hnsw" and it == "off" and k > int(pm["hnsw"]):
                    notes.append(f"pgvector HNSW: LIMIT {k} is above hnsw.ef_search ({pm['hnsw']}), "
                                 f"so it can return at most {pm['hnsw']} rows. Raise ef_search, or turn "
                                 "on iterative scan.")
            if cold:
                for o in outs:
                    cl = o["result"].get("clean") or {}
                    for er in cl.get("errors", []):
                        notes.append(f"{o['label']}: cache cleaning — {er}")
                    if cl.get("shared") is None and cl.get("os") is None and cl.get("cmd") is None:
                        notes.append(f"{o['label']}: nothing could be evicted — this 'cold' figure is not cold.")
                notes.append("Cold = the first run after eviction; warm = the median of the runs after it. "
                             "One cold sample is a single observation — use the sweep for a distribution.")
            if cat not in (None, "", "any"):
                notes.append(f"Filtered to category_id = {int(cat)}. Compare how many rows each engine "
                             f"returned out of {k}: a post-scan filter can come back short, and "
                             "pgvector's iterative scan or vchordrq's prefilter are what change that.")
            return jsonify({"ok": True, "label": label, "k": k, "opclass": opclass,
                            "reps": reps, "force_index": force, "exact": exact,
                            "engines": outs, "common_to_all": sorted(common), "notes": notes,
                            "protocol": {"cache_mode": "cold" if cold else "warm", "reps": reps, "jit": "off"}})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    def sample_queries(cur, table, nq, seed=7):
        cur.execute(f"SELECT min(id), max(id) FROM {qual(table)}")
        lo, hi = cur.fetchone()
        rng = np.random.default_rng(seed)
        ids = sorted({int(x) for x in rng.integers(int(lo), int(hi) + 1, size=nq * 3)})
        cur.execute(f"SELECT embedding::text FROM {qual(table)} WHERE id = ANY(%s) LIMIT %s",
                    (ids, nq))
        qvs = []
        for (e,) in cur.fetchall():
            v = np.array([float(x) for x in e.strip("[]").split(",")], dtype=np.float32)
            v = v + rng.standard_normal(v.shape[0]).astype(np.float32) * 0.05
            n = np.linalg.norm(v)
            qvs.append(v / n if n > 0 else v)
        return qvs

    def default_vc_probes(lists):
        if not lists:
            return [""]
        if len(lists) > 1:
            return [f"{p},{max(1, lists[1] // 100)}" for p in (1, 2, 4, 8, 16, 32)]
        cap = lists[0]
        pts = [p for p in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if p <= cap]
        return [str(p) for p in pts] or ["1"]

    def default_ef_list(k):
        out = [v for v in (10, 20, 40, 80, 160, 320, 640) if v >= k]
        return out or [max(k, 10)]

    def default_ivf_probes(lists):
        top = lists[0] if lists else 1
        return [p for p in (1, 2, 4, 8, 16, 32, 64, 128, 256) if p <= max(top, 1)] or [1]

    @app.route("/api/cmp/recall", methods=["POST"])
    def cmp_recall():
        """
        The measurement protocol, since latency at this scale is dominated by cache
        state and ordering rather than by the index:

          * every engine's setting i is measured before any engine's setting i+1,
            and the engine order rotates, so no engine is always first or always
            benefits from another's reads;
          * each setting is measured over `passes` full passes; a query's latency
            is the median of its passes, which discards one-off stalls;
          * warm mode prewarms the indexes and does a throwaway query per setting;
            cold mode evicts the engine's table and index from every reachable
            cache layer before EVERY pass;
          * recall carries a standard error, latency a pass-to-pass deviation, so
            two numbers that differ by less than their noise are visibly a tie.
        """
        body = request.get_json(silent=True) or {}
        conn = None
        try:
            es = need_engines(body.get("engines"))
            opclass = body.get("opclass", "vector_l2_ops")
            if opclass not in G["OPCLASSES"]:
                raise ValueError("Unknown opclass")
            k = max(1, min(200, int(body.get("k", 10))))
            nq = max(1, min(500, int(body.get("queries", 20))))
            force = bool(body.get("force_index", True))
            eps = body.get("epsilon")
            it = body.get("iterative") or "off"
            cold = body.get("cache_mode") == "cold"
            passes = max(1, min(9, int(body.get("passes") or (1 if cold else 3))))

            idx_of = {e["key"]: engine_indexes(e) for e in es}
            lists_of = lambda key: (idx_of[key][0].get("lists") if idx_of.get(key) else None)
            given = {"ivfplus": body.get("vp_probes_list"), "vchordrq": body.get("vc_probes_list"),
                     "hnsw": body.get("hnsw_ef_list"), "ivfflat": body.get("ivf_probes_list")}
            sweeps = {}
            for e in es:
                g, key = given.get(e["key"]), e["key"]
                if key == "ivfplus":
                    sweeps[key] = sorted({max(1, min(32768, int(p))) for p in
                                          (g or [1, 2, 4, 8, 16, 32, 64, 128])})
                elif key == "vchordrq":
                    sweeps[key] = [probes_ok(p) for p in (g or default_vc_probes(lists_of(key)))]
                    for p in sweeps[key]:
                        check_vc_probes(lists_of(key), p)
                elif key == "hnsw":
                    sweeps[key] = sorted({max(1, min(1000, int(p))) for p in (g or default_ef_list(k))})
                else:
                    sweeps[key] = sorted({max(1, int(p)) for p in (g or default_ivf_probes(lists_of(key)))})

            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.execute("SET jit = off")                 # JIT compile time is noise at this scale
            caps = cache_caps(cur)
            if cold:
                require_cold(caps)
            truth_table = truth_engine(es)["table"]
            qvs = sample_queries(cur, truth_table, nq)
            if not qvs:
                raise ValueError("Could not sample query vectors")
            nq = len(qvs)
            lits = [vec_literal(v) for v in qvs]

            def setters_for(e, p):
                return engine_setters(e["key"], p, eps if e["key"] == "vchordrq" else None,
                                      None, it if e["key"] in ("hnsw", "ivfflat") else None)

            def run_pass(qsql, setters, subset=None):
                set_scan_mode(cur, False, force)
                for s in setters:
                    cur.execute(s)
                times, got = [], []
                for lit in (lits if subset is None else lits[:subset]):
                    t0 = time.perf_counter()
                    cur.execute(qsql, (lit, lit))
                    ids = [r[0] for r in cur.fetchall()]
                    times.append((time.perf_counter() - t0) * 1000.0)
                    got.append(ids)
                return times, got

            # ground truth once, on a table that is not vchordrq's
            tsql = ann_sql(truth_table, opclass, k, None)
            set_scan_mode(cur, True, force)
            truth, exact_ms = [], []
            for lit in lits:
                t0 = time.perf_counter()
                cur.execute(tsql, (lit, lit))
                truth.append({r[0] for r in cur.fetchall()})
                exact_ms.append((time.perf_counter() - t0) * 1000.0)
            exact_avg = round(float(np.median(exact_ms)), 2)

            qsqls = {e["key"]: ann_sql(e["table"], opclass, k, None) for e in es}
            plans = {}
            for e in es:
                set_scan_mode(cur, False, force)
                for s in setters_for(e, sweeps[e["key"]][0]):
                    cur.execute(s)
                plans[e["key"]] = plan_of(cur, qsqls[e["key"]], (lits[0], lits[0]))

            prewarmed = {} if cold else {e["key"]: prewarm(cur, e) for e in es}

            # interleave: setting i of every engine before setting i+1 of any, rotating who goes first
            schedule = []
            for i in range(max(len(v) for v in sweeps.values())):
                row = [e for e in es if i < len(sweeps[e["key"]])]
                r = i % len(row)
                schedule += [(e, i) for e in row[r:] + row[:r]]

            results = {e["key"]: [None] * len(sweeps[e["key"]]) for e in es}
            cleans = {}
            for e, i in schedule:
                key, p = e["key"], sweeps[e["key"]][i]
                setters, qsql = setters_for(e, p), qsqls[key]
                if not cold:
                    run_pass(qsql, setters, subset=min(3, nq))      # throwaway, untimed
                grid, rec = [], None
                for pi in range(passes):
                    if cold:
                        cs = clean_caches(cur, e["table"])
                        cleans.setdefault(key, cs)
                    times, got = run_pass(qsql, setters)
                    grid.append(times)
                    if pi == 0:
                        rec = [len(t & set(g)) / max(1, len(t)) for t, g in zip(truth, got)]
                        ret = [len(g) for g in got]
                arr = np.array(grid)                               # passes x queries
                per_q = np.median(arr, axis=0)                     # each query's median over passes
                pass_means = arr.mean(axis=1)
                mean_ms = float(per_q.mean())
                results[key][i] = {
                    "param": p, "recall": round(float(np.mean(rec)), 4),
                    "recall_se": round(float(np.std(rec, ddof=1) / math.sqrt(len(rec))), 4) if len(rec) > 1 else None,
                    "ms": round(mean_ms, 3),
                    "ms_sd": round(float(np.std(pass_means, ddof=1)), 3) if passes > 1 else None,
                    "p50_ms": round(float(np.percentile(per_q, 50)), 3),
                    "p95_ms": round(float(np.percentile(per_q, 95)), 3),
                    "first_ms": round(float(arr[0][0]), 3),
                    "avg_returned": round(sum(ret) / len(ret), 1),
                    "speedup": round(exact_avg / mean_ms, 1) if mean_ms > 0 else None}

            at_floor = {}
            for floor in (0.90, 0.95, 0.99):
                row = {}
                for e in es:
                    ok = [r for r in results[e["key"]] if r["recall"] >= floor]
                    row[e["key"]] = min(ok, key=lambda r: r["ms"]) if ok else None
                at_floor[str(floor)] = row

            proto = (f"{'cold' if cold else 'warm'} cache, {passes} pass{'es' if passes > 1 else ''}, "
                     "engines interleaved")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_n, dim = count_rows(truth_table), dim_of(truth_table)
            for e in es:
                bq = store_query("SELECT id FROM builds WHERE target = ? AND engine = ? "
                                 "ORDER BY id DESC LIMIT 1", (e["target"], e["am"]))
                bid = bq[0]["id"] if bq else None
                for r in results[e["key"]]:
                    first = str(r["param"]).split(",")[0]
                    extra = (f"probes='{r['param']}' epsilon={eps}; " if e["key"] == "vchordrq" else "") + \
                            (f"iterative_scan={it}; " if it != "off" and e["key"] in ("hnsw", "ivfflat") else "")
                    store_insert("recalls", ts=stamp, build_id=bid, target=e["target"],
                                 label=e["label"], engine=e["am"], index_name=plans[e["key"]]["index"],
                                 rows=rows_n, dim=dim, k=k, queries=nq, param=e["param"],
                                 value=int(first) if first.isdigit() else 0, recall=r["recall"],
                                 ms=r["ms"], p95_ms=r["p95_ms"], avg_returned=r["avg_returned"],
                                 exact_ms=exact_avg, speedup=r["speedup"], note=extra + proto)
            log_query("cmp-recall-sweep", tsql, settings=[f"k = {k}", f"{nq} query vectors", proto] +
                      [f"{e['param']} over {sweeps[e['key']]}" for e in es],
                      ms=exact_avg, rows=sum(len(v) for v in results.values()) * nq * passes,
                      label=f"{opclass} · exact baseline {exact_avg}ms")

            notes = [n for n in (vp_index_note(),) if n]
            for e in es:
                if not plans[e["key"]]["index"]:
                    notes.append(f"{e['label']}: no index was used by the planner "
                                 f"({', '.join(plans[e['key']]['nodes'])}) — that sweep is measuring a "
                                 "sequential scan.")
            if cold:
                for key, cs in cleans.items():
                    for er in cs["errors"]:
                        notes.append(f"{engines_by_key()[key]['label']}: cache cleaning — {er}")
                    if cs["shared"] is None and cs["os"] is None and cs["cmd"] is None:
                        notes.append(f"{engines_by_key()[key]['label']}: nothing could be evicted — "
                                     "this 'cold' run is not cold.")
                    elif cs["os"] is None and not caps["shared_evict"]:
                        notes.append("Only the OS page cache was cleared; shared_buffers still holds pages "
                                     "(pg_buffercache is needed to evict them).")
                    elif cs["shared"] is not None and cs["os"] is None:
                        notes.append("shared_buffers was cleared but the OS page cache was not (this app is "
                                     "not on the database host, or cannot read the data directory), so "
                                     "'cold' reads may still be served from RAM.")
            elif passes == 1:
                notes.append("One pass only — latency has no repeat to take a median over. Use 3 or more.")
            if nq < 50:
                notes.append(f"{nq} query vectors: recall standard errors are wide. 50 or more is recommended.")
            return jsonify({"ok": True, "k": k, "queries": nq, "opclass": opclass, "rows": rows_n,
                            "exact_ms": exact_avg, "order": [e["key"] for e in es],
                            "engines": {e["key"]: {"label": e["label"], "results": results[e["key"]],
                                                   "index": plans[e["key"]]["index"],
                                                   "lists": lists_of(e["key"]), "param": e["param"]}
                                        for e in es},
                            "at_floor": at_floor, "notes": notes,
                            "protocol": {"cache_mode": "cold" if cold else "warm", "passes": passes,
                                         "interleaved": True, "jit": "off", "prewarm": prewarmed,
                                         "clean": cleans, "text": proto,
                                         "caps": {k_: caps[k_] for k_ in ("same_host", "os_evict", "shared_evict",
                                                                          "drop_caches_cmd", "pg_major")}},
                            "note": (f"{proto}; recall averaged over {nq} sampled query vectors against one "
                                     "shared exact ground truth; ± is the standard error (recall) and the "
                                     "pass-to-pass deviation (latency).")})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    def engines_by_key():
        return {e["key"]: e for e in engines()}

    # ── identical churn on every table ──────────────────────────────────────
    def _copy_with_ids(cur, table, ids, rows):
        out = []
        for i, (topic, cat, title, body, emb) in zip(ids, rows):
            t = title.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            b = body.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            out.append(f"{i}\t{topic}\t{cat}\t{t}\t{b}\t{vec_literal(emb)}\n")
        buf = io.StringIO("".join(out))
        t0 = time.perf_counter()
        cur.copy_expert(f"COPY {qual(table)} (id, topic, category_id, title, body, embedding) "
                        "FROM STDIN", buf)
        return time.perf_counter() - t0

    def _cmp_churn_worker(ins, upd, dele):
        CMP_CHURN.update({"running": True, "phase": "starting", "done": 0,
                          "total": ins + upd + dele, "error": None, "result": None})
        conn = None
        try:
            es = need_engines(None, need_index=False)
            t = {e["key"]: {"insert": 0.0, "update": 0.0, "delete": 0.0, "vacuum": 0.0} for e in es}
            dim = dim_of(es[0]["table"])
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            rng = np.random.default_rng(int(time.time()) & 0xFFFF)
            pyrng = __import__("random").Random(int(time.time()) & 0xFFFF)
            cur.execute("SELECT greatest(" + ", ".join(
                f"(SELECT coalesce(max(id),0) FROM {qual(e['table'])})" for e in es) + ")")
            nxt = int(cur.fetchone()[0]) + 1

            if ins:
                CMP_CHURN["phase"] = "insert"
                left = ins
                while left > 0:
                    n = min(1000, left)
                    rows = G["make_batch"](rng, dim, n, pyrng=pyrng)
                    ids = list(range(nxt, nxt + n))
                    for e in es:
                        t[e["key"]]["insert"] += _copy_with_ids(cur, e["table"], ids, rows)
                    nxt += n; left -= n
                    CMP_CHURN["done"] += n
                for e in es:
                    cur.execute("SELECT setval(pg_get_serial_sequence(%s,'id'), %s, true)",
                                (f"{schema()}.{e['table']}", nxt - 1))

            def pick(n):
                cur.execute(f"SELECT id FROM {qual(es[0]['table'])} ORDER BY random() LIMIT %s", (n,))
                return [r[0] for r in cur.fetchall()]

            if upd:
                CMP_CHURN["phase"] = "update"
                ids = pick(upd)
                for e in es:
                    t0 = time.perf_counter()
                    cur.execute(f"UPDATE {qual(e['table'])} SET body = body || ' revised' "
                                "WHERE id = ANY(%s)", (ids,))
                    t[e["key"]]["update"] += time.perf_counter() - t0
                CMP_CHURN["done"] += upd

            if dele:
                CMP_CHURN["phase"] = "delete"
                ids = pick(dele)
                for e in es:
                    t0 = time.perf_counter()
                    cur.execute(f"DELETE FROM {qual(e['table'])} WHERE id = ANY(%s)", (ids,))
                    t[e["key"]]["delete"] += time.perf_counter() - t0
                CMP_CHURN["done"] += dele
                CMP_CHURN["phase"] = "vacuum"
                for e in es:
                    t0 = time.perf_counter()
                    cur.execute(f"VACUUM {qual(e['table'])}")
                    t[e["key"]]["vacuum"] += time.perf_counter() - t0

            rows_after, idx_bytes = {}, {}
            for e in es:
                cur.execute(f"SELECT count(*) FROM {qual(e['table'])}")
                rows_after[e["key"]] = cur.fetchone()[0]
                idx_bytes[e["key"]] = sum(i["bytes"] for i in engine_indexes(e))
            CMP_CHURN["result"] = {
                "order": [e["key"] for e in es], "labels": {e["key"]: e["label"] for e in es},
                "timings": {k_: {p: round(v, 3) for p, v in d.items()} for k_, d in t.items()},
                "rows": rows_after, "index_bytes": idx_bytes,
                "inserts": ins, "updates": upd, "deletes": dele}
            log_query("cmp-churn", f"identical churn: {ins} inserts, {upd} updates, {dele} deletes on "
                      + ", ".join(e["table"] for e in es),
                      ms=sum(sum(d.values()) for d in t.values()) * 1000)
            CMP_CHURN["phase"] = "done"
        except Exception as e:
            CMP_CHURN["error"] = str(e)
            CMP_CHURN["phase"] = "error"
        finally:
            CMP_CHURN["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/cmp/churn", methods=["POST"])
    def cmp_churn():
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        body = request.get_json(silent=True) or {}
        try:
            need_engines(None, need_index=False)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        ins = max(0, min(200_000, int(body.get("inserts", 5000))))
        upd = max(0, min(200_000, int(body.get("updates", 2000))))
        dele = max(0, min(200_000, int(body.get("deletes", 1000))))
        threading.Thread(target=_cmp_churn_worker, args=(ins, upd, dele), daemon=True).start()
        return jsonify({"ok": True, "inserts": ins, "updates": upd, "deletes": dele})

    @app.route("/api/cmp/churn/status")
    def cmp_churn_status():
        return jsonify(CMP_CHURN)

    @app.route("/api/cmp/categories")
    def cmp_categories():
        try:
            rows = q(f"SELECT category_id, count(*) AS n FROM {qual(vp_table())} "
                     f"GROUP BY 1 ORDER BY 1")
        except Exception:
            rows = []
        return jsonify({"ok": True, "rows": rows})

    # saved connection is applied last, so it overrides the environment defaults
    # but is itself overridden by explicit command-line flags (they become the
    # parser defaults after this runs, then argparse applies whatever was passed).
    load_saved()
    print(f"  [vc_engine] registered — VectorChord table '{VC_TABLE}', "
          f"connection source: {CONN_META['source']}")
