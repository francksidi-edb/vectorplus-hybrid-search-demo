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

Two tables, same rows:
    vp_docs  — the existing table, indexed with ivfplus   (edb_vectorplus)
    vc_docs  — a row-for-row clone, indexed with vchordrq (vchord)
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
            "elapsed": 0, "error": None, "result": None}
CMP_CHURN = {"running": False, "phase": "idle", "done": 0, "total": 0, "error": None,
             "result": None}
CONN_META = {"source": "defaults / environment / command line"}


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
                         ("VectorChord build", VC_BUILD), ("table clone", CLONE),
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

    def vc_indexes(table=None):
        table = table or VC_TABLE
        try:
            rows = q("""
                SELECT i.relname AS name, pg_relation_size(i.oid) AS bytes,
                       pg_get_indexdef(i.oid) AS indexdef, x.indisvalid AS valid
                FROM pg_class i
                JOIN pg_index x ON x.indexrelid = i.oid
                JOIN pg_class t ON t.oid = x.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_am am ON am.oid = i.relam
                WHERE n.nspname = %s AND t.relname = %s AND am.amname = 'vchordrq'
                ORDER BY i.relname""", (schema(), table))
        except Exception:
            return []
        for r in rows:
            d = r.get("indexdef") or ""
            r["size"] = human_bytes(r["bytes"])
            m = re.search(r"lists\s*=\s*\[([0-9,\s]*)\]", d)
            r["lists"] = ([int(x) for x in m.group(1).split(",") if x.strip()] if m else None)
            m = re.search(r"rerank_in_table\s*=\s*(true|false)", d)
            r["rerank_in_table"] = (m.group(1) == "true") if m else False
            m = re.search(r"residual_quantization\s*=\s*(true|false)", d)
            r["residual_quantization"] = (m.group(1) == "true") if m else None
            m = re.search(r"USING\s+vchordrq\s*\(\s*(?:\(.*?\)|\w+)\s+(\w+)\s*\)", d, re.S)
            r["opclass"] = m.group(1) if m else None
        return rows

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

    def _clone_worker(step=50_000):
        CLONE.update({"running": True, "phase": "preparing", "done": 0, "total": 0,
                      "started": time.time(), "elapsed": 0, "error": None, "result": None})
        conn = None
        try:
            vp, vc = vp_table(), VC_TABLE
            dim = dim_of(vp)
            if not dim:
                raise RuntimeError(f"{vp} does not exist or has no embedding column — "
                                   "create and load it first.")
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.execute(f"DROP TABLE IF EXISTS {qual(vc)} CASCADE")
            log_query("ddl", f"DROP TABLE IF EXISTS {qual(vc)} CASCADE")
            # Same shape as the Vector Plus table, including the stored tsvector,
            # so table sizes and insert costs are comparable like for like.
            cur.execute(f"""
                CREATE TABLE {qual(vc)} (
                    id          bigserial PRIMARY KEY,
                    topic       text     NOT NULL,
                    category_id int      NOT NULL,
                    title       text     NOT NULL,
                    body        text     NOT NULL,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    tsv         tsvector GENERATED ALWAYS AS
                                (to_tsvector('english', title || ' ' || body)) STORED,
                    embedding   vector({dim})
                )""")
            cur.execute(f"SELECT count(*), coalesce(min(id),0), coalesce(max(id),0) FROM {qual(vp)}")
            total, lo, hi = cur.fetchone()
            CLONE["total"] = int(total)
            CLONE["phase"] = "copying"
            t0 = time.perf_counter()
            sql = (f"INSERT INTO {qual(vc)} (id, topic, category_id, title, body, created_at, embedding)\n"
                   f"SELECT id, topic, category_id, title, body, created_at, embedding\n"
                   f"FROM {qual(vp)} WHERE id >= %s AND id < %s")
            at = lo
            while at <= hi:
                cur.execute(sql, (at, at + step))
                CLONE["done"] += cur.rowcount
                CLONE["elapsed"] = round(time.time() - CLONE["started"], 2)
                at += step
            copy_s = round(time.perf_counter() - t0, 3)
            log_query("clone", sql, ms=copy_s * 1000, rows=CLONE["done"], label=vc)
            cur.execute("SELECT setval(pg_get_serial_sequence(%s, 'id'), %s, true)",
                        (f'{schema()}.{VC_TABLE}', max(1, int(hi))))

            CLONE["phase"] = "indexing"
            t1 = time.perf_counter()
            cur.execute(f'CREATE INDEX "{VC_TABLE}_cat_idx" ON {qual(vc)} (category_id)')
            cat_s = round(time.perf_counter() - t1, 3)
            CLONE["phase"] = "analyzing"
            cur.execute(f"ANALYZE {qual(vc)}")
            res = {"rows": CLONE["done"], "dim": dim, "copy_seconds": copy_s,
                   "rows_per_sec": int(CLONE["done"] / copy_s) if copy_s else 0,
                   "category_index_seconds": cat_s}
            CLONE["result"] = res
            store_insert("loads", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         target=VC_TARGET, rows=CLONE["done"], dim=dim, seed=None,
                         seconds=copy_s, rows_per_sec=res["rows_per_sec"],
                         generator=f"clone of {vp}")
            CLONE["phase"] = "done"
        except Exception as e:
            CLONE["error"] = str(e)
            CLONE["phase"] = "error"
        finally:
            CLONE["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    @app.route("/api/vc/table/clone", methods=["POST"])
    def vc_table_clone():
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        if not exists(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} does not exist yet. "
                            "Create and load the Vector Plus table first."}), 400
        if not count_rows(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} is empty."}), 400
        if not scalar("SELECT 1 FROM pg_extension WHERE extname = 'vchord'", default=0):
            return jsonify({"ok": False, "error": "vchord is not installed in this database "
                            "— use 'CREATE EXTENSION vchord' first."}), 400
        threading.Thread(target=_clone_worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/vc/table/clone/status")
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

    def _vc_build_worker(sql, name, lists, opclass, concurrently, maint_mem, workers, note):
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

        name = (body.get("name") or "").strip()
        if name:
            if not re.fullmatch(r"[A-Za-z0-9_]{1,60}", name):
                return jsonify({"ok": False, "error": "Index name must be letters, digits and underscores"}), 400
        else:
            short = opclass.replace("vector_", "v").replace("halfvec_", "h").replace("_ops", "")
            tag = "0" if not lists else "_".join(str(x) for x in lists)
            base, taken, n = f"{VC_PREFIX}_vchordrq_l{tag}_{short}", {r["name"] for r in vc_indexes()}, 2
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

        note = " ".join(options.split())
        threading.Thread(target=_vc_build_worker,
                         args=(sql, name, lists, opclass, concurrently, maint, workers, note),
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
    #  COMPARISON — shared query machinery
    # ════════════════════════════════════════════════════════════════════════

    def probes_ok(s):
        s = str(s if s is not None else "").replace(" ", "")
        if not _PROBES_RE.match(s):
            raise ValueError("vchordrq.probes must be digits, comma-separated per level — "
                             "e.g. 10 or 20,200, or empty for lists = []")
        return s

    def engine_setters(engine, probes, epsilon, prefilter=None):
        """SET statements for one engine; values are validated, so safe to inline."""
        if engine == "vp":
            return [f"SET ivfplus.probes = {int(probes)}"]
        out = [f"SET vchordrq.probes = '{probes_ok(probes)}'"]
        if epsilon not in (None, ""):
            out.append(f"SET vchordrq.epsilon = {float(epsilon)}")
        if prefilter is not None:
            out.append(f"SET vchordrq.prefilter = {'on' if prefilter else 'off'}")
        return out

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
        if exact:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
            cur.execute("SET enable_seqscan = on")
        else:
            cur.execute("SET enable_indexscan = on")
            cur.execute("SET enable_bitmapscan = on")
            cur.execute(f"SET enable_seqscan = {'off' if force_index else 'on'}")

    def run_ann(cur, table, qv, opclass, limit, category_id, setters,
                exact=False, force_index=True, reps=1, warm=True, kind="cmp"):
        sql = ann_sql(table, opclass, limit, category_id)
        lit = vec_literal(qv)
        params = (lit, lit)
        set_scan_mode(cur, exact, force_index)
        if not exact:
            for s in setters:
                cur.execute(s)
        plan = plan_of(cur, sql, params)
        if warm and not exact:
            cur.execute(sql, params)
            cur.fetchall()
        times, rows = [], []
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            cur.execute(sql, params)
            rows = cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
        full = resolve_sql(cur, sql, params)
        ms = float(np.median(times))
        log_query(kind, sql, settings=[s.replace("SET ", "") for s in setters] if not exact
                  else ["enable_indexscan = off"], ms=ms, rows=len(rows), sql_full=full,
                  label=table)
        return {"rows": [{"id": r[0], "topic": r[1], "category_id": r[2], "title": r[3],
                          "distance": round(float(r[4]), 6)} for r in rows],
                "ms": round(ms, 3), "ms_all": [round(t, 3) for t in times],
                "plan": plan, "sql": sql, "table": table}

    def need_both():
        vp, vc = vp_table(), VC_TABLE
        if not exists(vp):
            raise ValueError(f"{vp} does not exist")
        if not exists(vc):
            raise ValueError(f"{vc} does not exist — clone it from the VectorChord tab first")
        if dim_of(vp) != dim_of(vc):
            raise ValueError(f"{vp} is {dim_of(vp)}-dimensional but {vc} is {dim_of(vc)} — re-clone")
        if count_rows(vp) != count_rows(vc):
            raise ValueError(f"The tables have different row counts ({count_rows(vp):,} vs "
                             f"{count_rows(vc):,}) — re-clone so both hold the same rows")
        return vp, vc

    def single_vc_index():
        """With two vchordrq indexes on one table the planner picks whichever it
        likes, and the probes setting only fits one of them. Make it unambiguous."""
        idx = vc_indexes()
        if not idx:
            raise ValueError(f"{VC_TABLE} has no vchordrq index — build one in the VectorChord tab")
        if len(idx) > 1:
            raise ValueError(f"{VC_TABLE} has {len(idx)} vchordrq indexes ({', '.join(i['name'] for i in idx)}). "
                             "The planner chooses between them, which makes the comparison ambiguous — "
                             "drop all but the one you want to measure.")
        return idx[0]

    def vp_index_note():
        idx = [r for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"]
        if len(idx) > 1:
            return (f"{vp_table()} has {len(idx)} ivfplus indexes; the planner chose between them "
                    f"({', '.join(i['name'] for i in idx)}). Drop the others for a clean comparison.")
        return None

    # ── summary ─────────────────────────────────────────────────────────────
    @app.route("/api/cmp/summary")
    def cmp_summary():
        out = {"ok": True, "engines": {}}
        for key, table, engine, am, target, label in (
                ("vp", vp_table(), "ivfplus", "ivfplus", G["TARGET_NAME"], G["TARGET_LABEL"]),
                ("vc", VC_TABLE, "vchordrq", "vchordrq", VC_TARGET, VC_LABEL)):
            e = {"table": table, "label": label, "engine": engine, "exists": exists(table)}
            if e["exists"]:
                e["rows"] = count_rows(table)
                e["dim"] = dim_of(table)
                e["heap_bytes"] = int(scalar("SELECT pg_table_size(%s::regclass)",
                                             (f"{schema()}.{table}",), default=0) or 0)
                idx = (vc_indexes(table) if key == "vc" else
                       [r for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"])
                for r in idx:
                    r["size"] = human_bytes(r["bytes"])
                e["indexes"] = idx
                last = store_query("SELECT * FROM builds WHERE target = ? AND engine = ? "
                                   "ORDER BY id DESC LIMIT 1", (target, engine))
                if last:
                    b = last[0]
                    e["last_build"] = {k: b.get(k) for k in
                                       ("index_name", "build_seconds", "rows_per_sec", "index_bytes",
                                        "bytes_per_row", "lists", "opclass", "rows", "ts")}
                    e["last_build"]["size"] = human_bytes(b.get("index_bytes"))
            out["engines"][key] = e
        a, b = out["engines"]["vp"].get("last_build"), out["engines"]["vc"].get("last_build")
        if a and b and a.get("build_seconds") and b.get("build_seconds"):
            out["build_ratio_vc_over_vp"] = round(b["build_seconds"] / a["build_seconds"], 2)
        if a and b and a.get("index_bytes") and b.get("index_bytes"):
            out["size_ratio_vc_over_vp"] = round(b["index_bytes"] / a["index_bytes"], 2)
        return jsonify(out)

    # ── one query, both engines ────────────────────────────────────────────
    @app.route("/api/cmp/search", methods=["POST"])
    def cmp_search():
        body = request.get_json(silent=True) or {}
        conn = None
        try:
            vp, vc = need_both()
            vc_idx_one = single_vc_index()
            qv, label = G["resolve_query_vector"](body)
            opclass = body.get("opclass", "vector_l2_ops")
            if opclass not in G["OPCLASSES"]:
                raise ValueError("Unknown opclass")
            k = max(1, min(200, int(body.get("limit", 10))))
            cat = body.get("category_id")
            reps = max(1, min(20, int(body.get("reps", 3))))
            force = bool(body.get("force_index", True))
            vp_set = engine_setters("vp", int(body.get("vp_probes", 10)), None)
            pre = body.get("prefilter")
            vc_set = engine_setters("vc", body.get("vc_probes", ""), body.get("epsilon"),
                                    None if pre is None else bool(pre))

            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 120000")
            exact = run_ann(cur, vp, qv, opclass, k, cat, [], exact=True, kind="cmp-exact")
            a = run_ann(cur, vp, qv, opclass, k, cat, vp_set, force_index=force, reps=reps, kind="cmp-ivfplus")
            b = run_ann(cur, vc, qv, opclass, k, cat, vc_set, force_index=force, reps=reps, kind="cmp-vchordrq")

            truth = {r["id"] for r in exact["rows"]}
            ia, ib = {r["id"] for r in a["rows"]}, {r["id"] for r in b["rows"]}
            rec = lambda s: round(len(truth & s) / len(truth), 4) if truth else None
            notes = [n for n in (vp_index_note(),) if n]
            vec_names = {i["name"] for i in vc_indexes()} | \
                        {i["name"] for i in G["ivfplus_indexes"]()}
            for nm, res in (("ivfplus", a), ("vchordrq", b)):
                used = res["plan"]["index"]
                if used and used not in vec_names:
                    notes.append(f"{nm}: the planner used {used}, not a vector index, so this "
                                 f"result is exact for the filter and says nothing about ANN "
                                 f"recall. Selective filters often go this way.")
                if not res["plan"]["index"]:
                    notes.append(f"{nm}: the planner did not use an index "
                                 f"({', '.join(res['plan']['nodes'])}). Recall will be 1.0 and the "
                                 f"latency is a sequential scan — build an index with opclass {opclass}.")
            if cat not in (None, "", "any"):
                notes.append(f"Filtered to category_id = {int(cat)}. Compare how many rows each engine "
                             f"returned out of {k}: a post-scan filter can come back short.")
            return jsonify({
                "ok": True, "label": label, "k": k, "opclass": opclass,
                "settings": {"ivfplus": vp_set, "vchordrq": vc_set,
                             "force_index": force, "reps": reps},
                "exact": exact, "ivfplus": a, "vchordrq": b,
                "recall": {"ivfplus": rec(ia), "vchordrq": rec(ib)},
                "returned": {"ivfplus": len(a["rows"]), "vchordrq": len(b["rows"]), "exact": len(exact["rows"])},
                "overlap_between_engines": sorted(ia & ib),
                "speedup_vs_exact": {
                    "ivfplus": round(exact["ms"] / a["ms"], 1) if a["ms"] > 0 else None,
                    "vchordrq": round(exact["ms"] / b["ms"], 1) if b["ms"] > 0 else None},
                "notes": notes})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    # ── recall sweep, both engines, one ground truth ────────────────────────
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

    @app.route("/api/cmp/recall", methods=["POST"])
    def cmp_recall():
        body = request.get_json(silent=True) or {}
        conn = None
        try:
            vp, vc = need_both()
            opclass = body.get("opclass", "vector_l2_ops")
            if opclass not in G["OPCLASSES"]:
                raise ValueError("Unknown opclass")
            k = max(1, min(200, int(body.get("k", 10))))
            nq = max(1, min(200, int(body.get("queries", 20))))
            force = bool(body.get("force_index", True))
            eps = body.get("epsilon")
            vp_list = sorted({max(1, min(32768, int(p))) for p in
                              (body.get("vp_probes_list") or [1, 2, 4, 8, 16, 32, 64, 128])})
            vc_one = single_vc_index()
            vc_idx = [vc_one]
            lists = vc_one["lists"]
            vc_list = [probes_ok(p) for p in (body.get("vc_probes_list") or default_vc_probes(lists))]

            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            qvs = sample_queries(cur, vp, nq)
            if not qvs:
                raise ValueError("Could not sample query vectors")
            nq = len(qvs)
            sql = ann_sql(vp, opclass, k, None)
            sql_vc = ann_sql(vc, opclass, k, None)

            def topk(table, qsql, v, setters, exact=False):
                set_scan_mode(cur, exact, force)
                if not exact:
                    for s in setters:
                        cur.execute(s)
                lit = vec_literal(v)
                t0 = time.perf_counter()
                cur.execute(qsql, (lit, lit))
                got = [r[0] for r in cur.fetchall()]
                return got, (time.perf_counter() - t0) * 1000.0

            truth, exact_ms = [], []
            for v in qvs:
                got, ms = topk(vp, sql, v, [], exact=True)
                truth.append(set(got)); exact_ms.append(ms)
            exact_avg = round(sum(exact_ms) / len(exact_ms), 2)

            # Which index each engine actually uses, checked once up front.
            probe_lit = vec_literal(qvs[0])
            set_scan_mode(cur, False, force)
            for s in engine_setters("vp", vp_list[0], None):
                cur.execute(s)
            plan_vp = plan_of(cur, sql, (probe_lit, probe_lit))
            for s in engine_setters("vc", vc_list[0], eps):
                cur.execute(s)
            plan_vc = plan_of(cur, sql_vc, (probe_lit, probe_lit))

            def sweep(table, qsql, engine, values):
                out = []
                for p in values:
                    setters = engine_setters(engine, p, eps if engine == "vc" else None)
                    # one throwaway pass so the first value is not charged for a cold cache
                    topk(table, qsql, qvs[0], setters)
                    rs, ms_l, ret = 0.0, [], 0
                    for v, t in zip(qvs, truth):
                        got, ms = topk(table, qsql, v, setters)
                        rs += len(t & set(got)) / max(1, len(t))
                        ms_l.append(ms); ret += len(got)
                    out.append({"param": p, "recall": round(rs / nq, 4),
                                "ms": round(float(np.mean(ms_l)), 3),
                                "p95_ms": round(float(np.percentile(ms_l, 95)), 3),
                                "avg_returned": round(ret / nq, 1),
                                "speedup": round(exact_avg / np.mean(ms_l), 1) if np.mean(ms_l) > 0 else None})
                return out

            res_vp = sweep(vp, sql, "vp", vp_list)
            res_vc = sweep(vc, sql_vc, "vc", vc_list)

            # What matters in practice: the cheapest setting that clears a recall bar.
            at_floor = {}
            for floor in (0.90, 0.95, 0.99):
                def best(res):
                    ok = [r for r in res if r["recall"] >= floor]
                    return min(ok, key=lambda r: r["ms"]) if ok else None
                at_floor[str(floor)] = {"ivfplus": best(res_vp), "vchordrq": best(res_vc)}

            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows_n, dim = count_rows(vp), dim_of(vp)

            def persist(results, target, label, engine, pname, index_name):
                bq = store_query("SELECT id FROM builds WHERE target = ? AND engine = ? "
                                 "ORDER BY id DESC LIMIT 1", (target, engine))
                bid = bq[0]["id"] if bq else None
                for r in results:
                    first = str(r["param"]).split(",")[0]
                    store_insert("recalls", ts=stamp, build_id=bid, target=target, label=label,
                                 engine=engine, index_name=index_name, rows=rows_n, dim=dim,
                                 k=k, queries=nq, param=pname,
                                 value=int(first) if first.isdigit() else 0,
                                 recall=r["recall"], ms=r["ms"], p95_ms=r["p95_ms"],
                                 avg_returned=r["avg_returned"], exact_ms=exact_avg,
                                 speedup=r["speedup"],
                                 note=(f"probes='{r['param']}' epsilon={eps}" if engine == "vchordrq" else None))
            persist(res_vp, G["TARGET_NAME"], G["TARGET_LABEL"], "ivfplus", "ivfplus.probes", plan_vp["index"])
            persist(res_vc, VC_TARGET, VC_LABEL, "vchordrq", "vchordrq.probes", plan_vc["index"])

            log_query("cmp-recall-sweep", sql, settings=[f"k = {k}", f"{nq} query vectors",
                      f"ivfplus.probes over {vp_list}", f"vchordrq.probes over {vc_list}"],
                      ms=exact_avg, rows=(len(res_vp) + len(res_vc)) * nq,
                      label=f"{opclass} · exact baseline {exact_avg}ms")
            notes = [n for n in (vp_index_note(),) if n]
            for nm, pl in (("ivfplus", plan_vp), ("vchordrq", plan_vc)):
                if not pl["index"]:
                    notes.append(f"{nm}: no index was used by the planner ({', '.join(pl['nodes'])}) "
                                 f"— that sweep is measuring a sequential scan.")
            return jsonify({"ok": True, "k": k, "queries": nq, "opclass": opclass,
                            "rows": rows_n, "exact_ms": exact_avg,
                            "ivfplus": {"results": res_vp, "index": plan_vp["index"]},
                            "vchordrq": {"results": res_vc, "index": plan_vc["index"],
                                         "lists": lists, "epsilon": eps},
                            "at_floor": at_floor, "notes": notes,
                            "note": (f"Averaged over {nq} sampled query vectors against one shared "
                                     f"exact ground truth. The reference recommends at least 50 queries.")})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    # ── identical churn on both tables ──────────────────────────────────────
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
        t = {"ivfplus": {"insert": 0.0, "update": 0.0, "delete": 0.0, "vacuum": 0.0},
             "vchordrq": {"insert": 0.0, "update": 0.0, "delete": 0.0, "vacuum": 0.0}}
        try:
            vp, vc = need_both()
            dim = dim_of(vp)
            conn = get_conn(autocommit=True)
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            rng = np.random.default_rng(int(time.time()) & 0xFFFF)
            pyrng = __import__("random").Random(int(time.time()) & 0xFFFF)
            cur.execute(f"SELECT greatest((SELECT coalesce(max(id),0) FROM {qual(vp)}), "
                        f"(SELECT coalesce(max(id),0) FROM {qual(vc)}))")
            nxt = int(cur.fetchone()[0]) + 1

            if ins:
                CMP_CHURN["phase"] = "insert"
                left = ins
                while left > 0:
                    n = min(1000, left)
                    rows = G["make_batch"](rng, dim, n, pyrng=pyrng)
                    ids = list(range(nxt, nxt + n))
                    t["ivfplus"]["insert"] += _copy_with_ids(cur, vp, ids, rows)
                    t["vchordrq"]["insert"] += _copy_with_ids(cur, vc, ids, rows)
                    nxt += n; left -= n
                    CMP_CHURN["done"] += n
                for tb in (vp, vc):
                    cur.execute("SELECT setval(pg_get_serial_sequence(%s,'id'), %s, true)",
                                (f"{schema()}.{tb}", nxt - 1))

            def pick(n):
                cur.execute(f"SELECT id FROM {qual(vp)} ORDER BY random() LIMIT %s", (n,))
                return [r[0] for r in cur.fetchall()]

            if upd:
                CMP_CHURN["phase"] = "update"
                ids = pick(upd)
                for key, tb in (("ivfplus", vp), ("vchordrq", vc)):
                    t0 = time.perf_counter()
                    cur.execute(f"UPDATE {qual(tb)} SET body = body || ' revised' WHERE id = ANY(%s)", (ids,))
                    t[key]["update"] += time.perf_counter() - t0
                CMP_CHURN["done"] += upd

            if dele:
                CMP_CHURN["phase"] = "delete"
                ids = pick(dele)
                for key, tb in (("ivfplus", vp), ("vchordrq", vc)):
                    t0 = time.perf_counter()
                    cur.execute(f"DELETE FROM {qual(tb)} WHERE id = ANY(%s)", (ids,))
                    t[key]["delete"] += time.perf_counter() - t0
                CMP_CHURN["done"] += dele
                CMP_CHURN["phase"] = "vacuum"
                for key, tb in (("ivfplus", vp), ("vchordrq", vc)):
                    t0 = time.perf_counter()
                    cur.execute(f"VACUUM {qual(tb)}")
                    t[key]["vacuum"] += time.perf_counter() - t0

            cur.execute("SELECT count(*) FROM " + qual(vp)); rv = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM " + qual(vc)); rc = cur.fetchone()[0]
            CMP_CHURN["result"] = {
                "timings": {e: {k: round(v, 3) for k, v in d.items()} for e, d in t.items()},
                "rows": {"ivfplus": rv, "vchordrq": rc},
                "inserts": ins, "updates": upd, "deletes": dele,
                "index_bytes": {
                    "ivfplus": sum(r["bytes"] for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"),
                    "vchordrq": sum(r["bytes"] for r in vc_indexes())}}
            log_query("cmp-churn", f"identical churn: {ins} inserts, {upd} updates, {dele} deletes "
                      f"on {vp} and {vc}", ms=sum(sum(d.values()) for d in t.values()) * 1000)
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
            need_both()
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
