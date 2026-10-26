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
from flask import g, jsonify, request, send_from_directory

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

# WarehousePG: a second, independent connection profile. Its tables hold the same
# rows as vp_docs (synced from Postgres), one table per engine, so each index is
# the only one on its table exactly as on the Postgres side.
WH_TARGET = "whpg"
WH_TABLES = {"ivfplus": os.environ.get("VP_WH_VP_TABLE", "wvp_docs"),
             "hnsw":    os.environ.get("VP_WH_PGH_TABLE", "wpgh_docs"),
             "ivfflat": os.environ.get("VP_WH_PGI_TABLE", "wpgi_docs")}
WH = {"enabled": False, "host": "localhost", "port": 5433, "user": "gpadmin", "password": "",
      "database": "postgres", "schema": "public",
      # applied to EVERY WarehousePG session, before anything else runs
      "session": ["optimizer = off"],
      "distribution": "DISTRIBUTED BY (id)", "segments": None}
WH_BUILD = {"running": False, "sql": "", "started": None, "elapsed": 0, "error": None,
            "phase": "idle", "result": None, "notices": []}
WH_SYNC  = {"running": False, "phase": "idle", "done": 0, "total": 0, "started": None,
            "elapsed": 0, "error": None, "result": None, "current": None}
SWEEP = {"running": False, "phase": "idle", "done": 0, "total": 0, "current": "", "started": None,
         "elapsed": 0, "error": None, "result": None, "cancel": False, "run_id": None}
# ClickHouse on the same machine: its connection, and an opt-in way to start / stop its service.
CH = {"enabled": False, "host": "localhost", "port": 8123, "user": "default", "password": "",
      "database": "default", "service": "clickhouse-server"}
CH_SVC = {"running": False, "action": None, "phase": "idle", "steps": [], "started": None, "elapsed": 0,
          "error": None, "hint": None, "result": None}
# Starting and stopping a system service from a web page is dangerous on an app that has no login and
# listens on every interface, so it is OFF unless the operator turns it on when starting the app.
SERVICE_CONTROL = os.environ.get("VP_ALLOW_SERVICE_CONTROL", "0").lower() in ("1", "true", "yes")
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")
CH_TABLE  = os.environ.get("VP_CH_TABLE", "ch_docs")
CH_TARGET = "clickhouse"
CHD_SYNC  = {"running": False, "phase": "idle", "done": 0, "total": 0, "started": None, "elapsed": 0,
             "error": None, "result": None}
CHQ_TABLE = os.environ.get("VP_CHQ_TABLE", "chq_docs")        # the same rows with the embedding stored as a QBit column
CHQ_BUILD = {"running": False, "phase": "idle", "started": None, "elapsed": 0, "error": None, "result": None, "sql": ""}
CHD_BUILD = {"running": False, "phase": "idle", "started": None, "elapsed": 0, "error": None,
             "result": None, "sql": "", "parts_to_do": None}
SEARCH = {"running": False, "phase": "idle", "done": 0, "total": 0, "current": "", "started": None, "elapsed": 0,
          "error": None, "result": None, "cancel": False}
_SETTING_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*(?:=|\sTO\s)\s*([A-Za-z0-9_.,'\- ]{1,60})$", re.I)
_DIST_RE = re.compile(r"^(DISTRIBUTED\s+(BY\s*\(\s*[A-Za-z_]\w*(\s*,\s*[A-Za-z_]\w*)*\s*\)|RANDOMLY|REPLICATED))?$", re.I)


def register(app, G):
    """Attach every route. G is the host module's globals()."""

    # The host serves vectorplus_dashboard.html from the folder that holds vectorplus_api.py. In the repository the
    # page lives in dashboard/, so when the sibling copy is missing, serve the one in dashboard/ instead.
    # (vectorplus_api.py is not touched; when a sibling copy exists the host's own route is used as before.)
    _here = Path(G.get("__file__", __file__)).resolve().parent

    @app.before_request
    def _dashboard_from_subfolder():
        if request.method in ("GET", "HEAD") and request.path in ("/", "/vectorplus_dashboard.html"):
            if not (_here / "vectorplus_dashboard.html").exists() and (_here / "dashboard" / "vectorplus_dashboard.html").exists():
                return send_from_directory(str(_here / "dashboard"), "vectorplus_dashboard.html")
        return None

    # ── thin wrappers over the host's helpers ───────────────────────────────
    def schema(prof="pg"):     return G["SCHEMA"] if prof == "pg" else (WH["schema"] if prof == "whpg" else CH["database"])
    def qual(name, prof="pg"):
        if prof == "pg":
            return G["qualified"](name)
        if prof == "ch":
            return f"`{CH['database']}`.`{name}`"
        return f'"{WH["schema"]}"."{name}"'
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

    def exists(table, prof="pg"):
        if prof == "ch":
            try:
                return int(ch_rows(f"SELECT count() AS n FROM system.tables WHERE database = '{CH['database']}' AND name = '{table}'")[0]["n"]) > 0
            except Exception:
                return False
        return bool(pscalar(prof, "SELECT to_regclass(%s) IS NOT NULL",
                            (f"{schema(prof)}.{table}",), default=False))

    def dim_of(table, prof="pg"):
        if prof == "ch" and table == CHQ_TABLE:
            try:
                r_ = ch_rows(f"SELECT type FROM system.columns WHERE database = '{CH['database']}' AND table = '{table}' AND name = 'embedding'")
                m_ = re.search(r"QBit\(\w+,\s*(\d+)\)", r_[0]["type"]) if r_ else None
                return int(m_.group(1)) if m_ else 0
            except Exception:
                return 0
        if prof == "ch":
            try:
                r_ = ch_rows(f"SELECT length(embedding) AS d FROM {qual(table, 'ch')} LIMIT 1")
                return int(r_[0]["d"]) if r_ else 0
            except Exception:
                return 0
        v = pscalar(prof, """SELECT a.atttypmod FROM pg_attribute a
                      JOIN pg_class c ON c.oid = a.attrelid
                      JOIN pg_namespace n ON n.oid = c.relnamespace
                      WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding'""",
                    (schema(prof), table), default=0)
        return int(v) if v and v > 0 else 0

    def count_rows(table, prof="pg"):
        return int(pscalar(prof, f"SELECT count(*) FROM {qual(table, prof)}", default=0) or 0)

    # ── connection profiles ─────────────────────────────────────────────────
    # "pg" is the host app's own connection. "whpg" is WarehousePG: a separate
    # host/port/credentials, and a list of session settings (optimizer = off by
    # default) that is applied to every connection opened for it.
    def wh_settings_sql(settings=None):
        out = []
        for line in (WH.get("session") if settings is None else settings) or []:
            line = str(line).strip()
            if not line:
                continue
            m = _SETTING_RE.match(line)
            if not m:
                raise ValueError(f"Bad session setting {line!r} — write it as  name = value")
            out.append(f"SET {m.group(1)} = {m.group(2).strip()}")
        return out

    def wh_connect(cfg=None, autocommit=False, settings=None):
        cfg = cfg or WH
        conn = psycopg2.connect(host=cfg["host"], port=int(cfg["port"]), user=cfg["user"],
                                password=cfg.get("password", ""), dbname=cfg["database"],
                                connect_timeout=8)
        conn.autocommit = True               # session settings must not sit inside a transaction
        cur = conn.cursor()
        for stmt in wh_settings_sql(settings):
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.close()
                raise RuntimeError(f"WarehousePG session setting failed — {stmt}: "
                                   f"{str(e).strip().splitlines()[0]}")
        conn.autocommit = autocommit
        return conn

    def prof_conn(prof, autocommit=False):
        if prof == "ch":
            return ChConn()
        return get_conn(autocommit=autocommit) if prof == "pg" else wh_connect(autocommit=autocommit)

    def wh_error():
        """None when WarehousePG is off or reachable with the current profile, else why not."""
        if not WH["enabled"]:
            return None
        try:
            wh_connect().close()
            return None
        except Exception as e:
            return str(e).strip().splitlines()[0]

    def wh_down_message(err):
        return (f"WarehousePG is enabled but cannot be reached: {err}. If the password was not saved "
                "to the file, re-enter it under Setup → Connection → WarehousePG and press Apply.")

    def pq(prof, sql, params=None):
        """Run a read on the named profile; rows come back as dicts."""
        if prof == "pg":
            return q(sql, params)
        if prof == "ch":
            return ch_rows(sql)
        conn = wh_connect()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            conn.commit()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def pscalar(prof, sql, params=None, default=None):
        try:
            r = pq(prof, sql, params)
            return list(r[0].values())[0] if r else default
        except Exception:
            return default

    def soft(cur, sql):
        """Execute a setting that some servers do not know; never fail the run for it."""
        try:
            cur.execute(sql)
            return True
        except Exception:
            return False

    class Conns:
        """One autocommit connection per profile, opened lazily and closed together."""
        def __init__(self, timeout_ms=0, registry=None):
            self.conn, self.cur, self.timeout, self.registry = {}, {}, timeout_ms, registry
        def cursor(self, prof):
            if prof not in self.conn:
                self.conn[prof] = prof_conn(prof, autocommit=True)
                c = self.conn[prof].cursor()
                soft(c, f"SET statement_timeout = {int(self.timeout)}")
                soft(c, "SET jit = off")
                if self.registry is not None:
                    try:
                        c.execute("SELECT pg_backend_pid()")
                        self.registry[prof] = c.fetchone()[0]
                    except Exception:
                        pass
                self.cur[prof] = c
            return self.cur[prof]
        def close(self):
            for c in self.conn.values():
                try: c.close()
                except Exception: pass

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
        load_saved_wh()
        load_saved_ch()

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
                         ("WarehousePG build", WH_BUILD), ("WarehousePG sync", WH_SYNC), ("recall sweep", SWEEP), ("ClickHouse service action", CH_SVC),
                         ("one-query search", SEARCH), ("ClickHouse sync", CHD_SYNC), ("ClickHouse index build", CHD_BUILD), ("ClickHouse QBit build", CHQ_BUILD),
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
            payload["password"] = cfg["password"] if body.get("save_password") else None
            try:
                write_saved(payload)
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

    # ── WarehousePG connection (second profile) ─────────────────────────────
    def write_saved(update):
        """Merge keys into the saved-connection file (mode 600), keeping what is there."""
        try:
            with open(CONN_FILE) as fh:
                cur = json.load(fh)
        except Exception:
            cur = {}
        cur.update(update)
        fd = os.open(CONN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(cur, fh, indent=2)

    def wh_public():
        return {"enabled": WH["enabled"], "host": WH["host"], "port": WH["port"], "user": WH["user"],
                "dbname": WH["database"], "schema": WH["schema"], "has_password": bool(WH["password"]),
                "session": list(WH["session"]), "distribution": WH["distribution"],
                "segments": WH["segments"], "tables": dict(WH_TABLES),
                "saved_file": CONN_FILE, "saved_exists": os.path.exists(CONN_FILE)}

    def wh_candidate(body):
        host = (body.get("host") or "").strip()
        user = (body.get("user") or "").strip()
        dbn = (body.get("dbname") or "").strip()
        sch = (body.get("schema") or "public").strip()
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
            pw = WH.get("password", "")
        sess = body.get("session")
        if isinstance(sess, str):
            sess = sess.splitlines()
        sess = [s.strip() for s in (sess if sess is not None else WH["session"]) if str(s).strip()]
        wh_settings_sql(sess)                                  # validates the syntax
        dist = (body.get("distribution") if body.get("distribution") is not None
                else WH["distribution"]).strip()
        if not _DIST_RE.match(dist):
            raise ValueError("distribution must be empty, DISTRIBUTED BY (col, …), DISTRIBUTED RANDOMLY "
                             "or DISTRIBUTED REPLICATED")
        return {"host": host, "port": port, "user": user, "password": pw, "database": dbn,
                "schema": sch, "session": sess, "distribution": dist}

    def wh_probe(cfg):
        """Connect with the candidate settings and report what is there."""
        t0 = time.perf_counter()
        conn = wh_connect(cfg, autocommit=True, settings=cfg["session"])
        try:
            ms = (time.perf_counter() - t0) * 1000
            cur = conn.cursor()
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
            cur.execute("SELECT extname, extversion FROM pg_extension "
                        "WHERE extname IN ('edb_vectorplus','vector')")
            exts = {n: v for n, v in cur.fetchall()}
            seg = None
            try:
                cur.execute("SELECT count(*) FROM gp_segment_configuration WHERE role = 'p' AND content >= 0")
                seg = int(cur.fetchone()[0])
            except Exception:
                pass
            applied = {}
            for stmt in wh_settings_sql(cfg["session"]):
                name = stmt.split()[1]
                try:
                    cur.execute(f"SHOW {name}")
                    applied[name] = cur.fetchone()[0]
                except Exception:
                    applied[name] = None
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)", (cfg["schema"],))
            sch_ok = cur.fetchone()[0]
            return {"version": ver, "connect_ms": round(ms, 1), "extensions": exts,
                    "segments": seg, "applied": applied, "schema_exists": sch_ok}
        finally:
            conn.close()

    @app.route("/api/conn/whpg")
    def whpg_get():
        return jsonify({"ok": True, **wh_public()})

    @app.route("/api/conn/whpg/test", methods=["POST"])
    def whpg_test():
        try:
            cfg = wh_candidate(request.get_json(silent=True) or {})
            return jsonify({"ok": True, **wh_probe(cfg)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e).strip()}), 400

    @app.route("/api/conn/whpg/apply", methods=["POST"])
    def whpg_apply():
        body = request.get_json(silent=True) or {}
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        try:
            enabled = bool(body.get("enabled", True))
            cfg = wh_candidate(body)
            info = wh_probe(cfg) if enabled else {}
        except Exception as e:
            return jsonify({"ok": False, "error": str(e).strip()}), 400
        WH.update(cfg)
        WH["enabled"] = enabled
        WH["segments"] = info.get("segments")
        saved = False
        if body.get("save"):
            payload = {k: cfg[k] for k in ("host", "port", "user", "database", "schema", "session", "distribution")}
            payload["enabled"] = enabled
            if body.get("save_password"):
                payload["password"] = cfg["password"]
            try:
                write_saved({"whpg": payload})
                saved = True
            except Exception as e:
                info["save_error"] = str(e)
        return jsonify({"ok": True, "saved": saved, **info, **wh_public()})

    def _first_line(e):
        return str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__

    def _product(version):
        import re as _re
        m = _re.search(r"(Greenplum Database [^\s]+(?:-WHPG)?)", version or "")
        if m:
            return m.group(1)
        m = _re.match(r"(PostgreSQL [^\s,]+)", version or "")
        return m.group(1) if m else (version or "")[:40]

    @app.route("/api/conn/ping")
    def conn_ping():
        """Opens each connection for real and says so. This, not the saved settings, is what the
        'connected' badges reflect."""
        out = {"ok": True, "ts": datetime.now().strftime("%H:%M:%S"),
               "pg": {"ok": False}, "whpg": {"enabled": WH["enabled"], "ok": False}}
        c = conn_public()
        out["pg"].update(target=f"{c['user']}@{c['host']}:{c['port']}/{c['dbname']}", schema=c["schema"])
        t0 = time.perf_counter()
        try:
            conn = get_conn()
            try:
                cur = conn.cursor()
                cur.execute("SELECT version(), current_setting('server_version')")
                ver, sv = cur.fetchone()
            finally:
                conn.close()
            out["pg"].update(ok=True, ms=round((time.perf_counter() - t0) * 1000, 1), version=sv,
                             product=_product(ver))
        except Exception as e:
            out["pg"]["error"] = _first_line(e)
        w = out["whpg"]
        w["target"] = f"{WH['user']}@{WH['host']}:{WH['port']}/{WH['database']}"
        if WH["enabled"]:
            t0 = time.perf_counter()
            try:
                conn = wh_connect()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT version()")
                    ver = cur.fetchone()[0]
                finally:
                    conn.close()
                w.update(ok=True, ms=round((time.perf_counter() - t0) * 1000, 1), product=_product(ver),
                         segments=WH.get("segments"), session=list(WH["session"]))
            except Exception as e:
                w["error"] = _first_line(e)
        c_ = {"enabled": CH["enabled"], "ok": False, "paused": bool(CH.get("quiet")), "target": f"{CH['user']}@{CH['host']}:{CH['port']}/{CH['database']}"}
        ss_ = ch_service_state(CH)
        c_["service"], c_["service_note"] = ss_.get("state"), ss_.get("note")
        if CH["enabled"] and not CH.get("quiet"):
            try:
                txt_, ms_ = ch_ping(CH, 3)
                c_.update(ok=(txt_ == "Ok."), ms=ms_, version=ch_query(CH, "SELECT version()", timeout=3).strip())
            except Exception as e:
                c_["error"] = _first_line(e).replace("<urlopen error ", "").rstrip(">")
        out["ch"] = c_
        return jsonify(out)

    @app.route("/api/conn/whpg/disconnect", methods=["POST"])
    def whpg_disconnect():
        """Stop using WarehousePG. The settings stay filled in, so Connect brings it straight back."""
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        WH["enabled"] = False
        try:
            with open(CONN_FILE) as fh:
                saved = json.load(fh)
            if isinstance(saved.get("whpg"), dict):
                saved["whpg"]["enabled"] = False
                write_saved({"whpg": saved["whpg"]})
        except Exception:
            pass
        return jsonify({"ok": True, **wh_public()})

    def load_saved_wh():
        try:
            with open(CONN_FILE) as fh:
                s = json.load(fh).get("whpg")
        except Exception:
            return
        if not s:
            return
        for k in ("host", "user", "database", "schema", "distribution"):
            if s.get(k) is not None:
                WH[k] = s[k]
        if s.get("port"):
            WH["port"] = int(s["port"])
        if s.get("password"):
            WH["password"] = s["password"]
        if isinstance(s.get("session"), list):
            WH["session"] = s["session"]
        WH["enabled"] = bool(s.get("enabled"))

    # ════════════════════════════════════════════════════════════════════════
    #  VECTORCHORD — status, extension, table clone
    # ════════════════════════════════════════════════════════════════════════

    def am_indexes(table, am, prof="pg"):
        """Indexes of one access method on one table, with the options parsed out of
        pg_get_indexdef so the UI can show what each was built with."""
        try:
            rows = pq(prof, """
                SELECT i.relname AS name, pg_relation_size(i.oid) AS bytes,
                       pg_get_indexdef(i.oid) AS indexdef, x.indisvalid AS valid
                FROM pg_class i
                JOIN pg_index x ON x.indexrelid = i.oid
                JOIN pg_class t ON t.oid = x.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_am am ON am.oid = i.relam
                WHERE n.nspname = %s AND t.relname = %s AND am.amname = %s
                ORDER BY i.relname""", (schema(prof), table, am))
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
            # Postgres leaves the opclass out of pg_get_indexdef when it is the access method's default
            r["opclass"] = m.group(1) if m else ("vector_l2_ops (default)" if am != "hnsw" else None)
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
        bv = bc = None
        if out["vp_exists"]:
            bv = brief(vp)
            out["vp_rows"], out["vp_rows_exact"], out["vp_max_id"], out["dim"] = bv["rows"], bv["exact"], bv["max_id"], dim_of(vp)
        if out["vc_exists"]:
            bc = brief(vc)
            out["vc_rows"], out["vc_rows_exact"], out["vc_max_id"], out["vc_dim"] = bc["rows"], bc["exact"], bc["max_id"], dim_of(vc)
            out["vc_size"] = scalar("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                    (f"{schema()}.{vc}",))
            out["indexes"] = vc_indexes()
        if out["vp_exists"] and out["vc_exists"]:
            out["in_sync"] = sync_state(bv, bc)
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
        fp_invalidate()
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
                CLONE["phase"] = "vacuum + analyze"
                cur.execute(f"VACUUM (ANALYZE) {qual(tb)}")
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
            fp_invalidate()
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
        vp_rows, bvp = None, None
        if out["vp_exists"]:
            bvp = brief(vp_table())
            vp_rows = bvp["rows"]
            out["vp_rows"], out["vp_rows_exact"], out["dim"] = vp_rows, bvp["exact"], dim_of(vp_table())
        out["tables"] = {}
        for am, tb in pgv_tables().items():
            t = {"table": tb, "exists": exists(tb), "am": am}
            if t["exists"]:
                bt = brief(tb)
                t["rows"], t["rows_exact"] = bt["rows"], bt["exact"]
                t["size"] = scalar("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                   (f"{schema()}.{tb}",))
                t["indexes"] = am_indexes(tb, am)
                if bvp is not None:
                    t["in_sync"] = sync_state(bvp, bt)
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
    #  WAREHOUSEPG — tables synced from Postgres, indexes built with matching parameters
    # ════════════════════════════════════════════════════════════════════════
    # No CREATE INDEX CONCURRENTLY anywhere in this section: WarehousePG does not
    # support it, so it is never offered here. Every connection runs the profile's
    # session settings first (optimizer = off by default).

    WH_LABEL = {"ivfplus": "EDB ivfplus", "hnsw": "pgvector HNSW", "ivfflat": "pgvector IVFFlat"}

    def wh_table_ddl(table, dim):
        dist = WH.get("distribution") or ""
        return f"""
            CREATE TABLE {qual(table, "whpg")} (
                id          bigint PRIMARY KEY,
                topic       text   NOT NULL,
                category_id int    NOT NULL,
                title       text   NOT NULL,
                body        text   NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now(),
                embedding   vector({dim})
            ) {dist}"""

    # An exact fingerprint (rows, id range, id checksum) needs a full scan: seconds at 1M rows, tens
    # of seconds at 10M. It is therefore computed once and kept for FP_TTL, invalidated by every
    # operation that changes rows through this app. Polled status endpoints never wait for it: they
    # show the planner's row estimate and ask for the exact figure in the background.
    FP_CACHE, FP_BUSY, FP_TTL = {}, set(), 900
    FP_LOCK, FP_STATE = threading.Lock(), {"epoch": 0}

    def fp_invalidate():
        with FP_LOCK:
            FP_CACHE.clear()
            FP_STATE["epoch"] += 1

    def _fp_compute(table, prof):
        r = pq(prof, f"SELECT count(*) AS n, coalesce(min(id),0) AS lo, coalesce(max(id),0) AS hi, "
                     f"coalesce(sum(id),0)::text AS s FROM {qual(table, prof)}")
        return (int(r[0]["n"]), int(r[0]["lo"]), int(r[0]["hi"]), r[0]["s"]) if r else None

    def fingerprint(table, prof="pg", fresh=False):
        """Exact, cached. May take a while the first time; comparison runs use this."""
        key = (prof, table)
        with FP_LOCK:
            hit, epoch = FP_CACHE.get(key), FP_STATE["epoch"]
        if hit and not fresh and time.time() - hit[1] < FP_TTL:
            return hit[0]
        fp = _fp_compute(table, prof)
        with FP_LOCK:
            if FP_STATE["epoch"] == epoch:
                FP_CACHE[key] = (fp, time.time())
        return fp

    def fingerprint_cached(table, prof="pg"):
        """For polled status: the exact fingerprint if known, else None — and one background scan is started."""
        key = (prof, table)
        with FP_LOCK:
            hit = FP_CACHE.get(key)
            if hit and time.time() - hit[1] < FP_TTL:
                return hit[0]
            if key in FP_BUSY:
                return None
            FP_BUSY.add(key)
            epoch = FP_STATE["epoch"]

        def work():
            try:
                fp = _fp_compute(table, prof)
                with FP_LOCK:
                    if FP_STATE["epoch"] == epoch:
                        FP_CACHE[key] = (fp, time.time())
            except Exception:
                pass
            finally:
                with FP_LOCK:
                    FP_BUSY.discard(key)
        threading.Thread(target=work, daemon=True).start()
        return None

    def rows_est(table, prof="pg"):
        if prof == "ch":
            try:
                r_ = ch_rows(f"SELECT total_rows AS n FROM system.tables WHERE database = '{CH['database']}' AND name = '{table}'")
                return int(r_[0]["n"]) if r_ and r_[0]["n"] else None
            except Exception:
                return None
        v = pscalar(prof, "SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(%s)",
                    (f"{schema(prof)}.{table}",))
        return int(v) if v is not None and int(v) > 0 else None

    def brief(table, prof="pg"):
        """Cheap view of a table for a polled endpoint: estimate, newest id (an index lookup) and the
        exact fingerprint when it is already known."""
        fp = fingerprint_cached(table, prof)
        return {"rows": fp[0] if fp else rows_est(table, prof), "exact": bool(fp),
                "max_id": pscalar(prof, f"SELECT max(id) FROM {qual(table, prof)}"), "fp": fp}

    def sync_state(a, b):
        """True / False when known; None while the exact check is still running."""
        if a["fp"] and b["fp"]:
            return a["fp"] == b["fp"]
        if a["max_id"] is not None and b["max_id"] is not None and a["max_id"] != b["max_id"]:
            return False
        return None

    @app.route("/api/wh/status")
    def wh_status():
        out = {"enabled": WH["enabled"], "config": wh_public(), "connected": False,
               "sync": WH_SYNC, "build": {k: v for k, v in WH_BUILD.items() if k != "notices"}}
        if not WH["enabled"]:
            return jsonify(out)
        try:
            info = wh_probe({k: WH[k] for k in ("host", "port", "user", "password", "database",
                                                "schema", "session")})
        except Exception as e:
            out["error"] = str(e)
            return jsonify(out)
        out.update(info, connected=True)
        WH["segments"] = info.get("segments")
        out["pg_exists"] = exists(vp_table())
        src = fingerprint_cached(vp_table()) if out["pg_exists"] else None
        out["source"] = {"table": vp_table(), "rows": src[0] if src else (rows_est(vp_table()) or 0),
                         "exact": bool(src)}
        out["tables"] = {}
        for am, tb in WH_TABLES.items():
            t = {"table": tb, "am": am, "exists": exists(tb, "whpg")}
            if t["exists"]:
                fp = fingerprint_cached(tb, "whpg")
                t["rows"], t["rows_exact"] = (fp[0] if fp else rows_est(tb, "whpg")), bool(fp)
                t["in_sync"] = (fp == src) if (fp and src) else None
                t["size"] = pscalar("whpg", "SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))",
                                    (f"{schema('whpg')}.{tb}",))
                t["indexes"] = am_indexes(tb, am, "whpg")
            out["tables"][am] = t
        return jsonify(out)

    def _wh_sync_worker(tables, step=50_000):
        fp_invalidate()
        WH_SYNC.update({"running": True, "phase": "preparing", "done": 0, "total": 0,
                        "started": time.time(), "elapsed": 0, "error": None, "result": None,
                        "current": None})
        pgc = wc = None
        try:
            vp = vp_table()
            dim = dim_of(vp)
            if not dim:
                raise RuntimeError(f"{vp} does not exist or has no embedding column")
            pgc = get_conn(autocommit=True)
            pcur = pgc.cursor()
            pcur.execute("SET statement_timeout = 0")
            pcur.execute(f"SELECT count(*), coalesce(min(id),0), coalesce(max(id),0) FROM {qual(vp)}")
            total, lo, hi = pcur.fetchone()
            WH_SYNC["total"] = int(total) * len(tables)
            wc = wh_connect(autocommit=True)
            wcur = wc.cursor()
            soft(wcur, "SET statement_timeout = 0")
            cols = "id, topic, category_id, title, body, created_at, embedding"
            first, binary, per = None, True, []
            for tb in tables:
                WH_SYNC["current"], WH_SYNC["phase"] = tb, "creating"
                wcur.execute(f"DROP TABLE IF EXISTS {qual(tb, 'whpg')}")
                wcur.execute(wh_table_ddl(tb, dim))
                t0, copied = time.perf_counter(), 0
                WH_SYNC["phase"] = "copying"
                if first is None:
                    # stream from Postgres, binary if both ends accept it, text otherwise
                    at = lo
                    while at <= hi:
                        sel = (f"SELECT {cols} FROM {qual(vp)} WHERE id >= {int(at)} AND id < {int(at) + step}")
                        for attempt in (0, 1):
                            fmt = "FORMAT BINARY" if binary else "FORMAT TEXT"
                            buf = io.BytesIO()
                            pcur.copy_expert(f"COPY ({sel}) TO STDOUT WITH ({fmt})", buf)
                            buf.seek(0)
                            try:
                                wcur.copy_expert(f"COPY {qual(tb, 'whpg')} ({cols}) FROM STDIN WITH ({fmt})", buf)
                                break
                            except Exception:
                                if binary and attempt == 0:
                                    binary = False          # this server refused binary — redo as text
                                    continue
                                raise
                        copied += max(0, wcur.rowcount)
                        WH_SYNC["done"] += max(0, wcur.rowcount)
                        WH_SYNC["elapsed"] = round(time.time() - WH_SYNC["started"], 2)
                        at += step
                    first = tb
                else:
                    wcur.execute(f"INSERT INTO {qual(tb, 'whpg')} ({cols}) "
                                 f"SELECT {cols} FROM {qual(first, 'whpg')}")
                    copied = max(0, wcur.rowcount)
                    WH_SYNC["done"] += copied
                secs = round(time.perf_counter() - t0, 3)
                WH_SYNC["phase"] = "indexing"
                wcur.execute(f'CREATE INDEX "{tb}_cat_idx" ON {qual(tb, "whpg")} (category_id)')
                WH_SYNC["phase"] = "analyzing"
                wcur.execute(f"ANALYZE {qual(tb, 'whpg')}")
                per.append({"table": tb, "rows": copied, "seconds": secs,
                            "rows_per_sec": int(copied / secs) if secs else 0,
                            "binary": binary})
                store_insert("loads", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target=WH_TARGET,
                             rows=copied, dim=dim, seed=None, seconds=secs,
                             rows_per_sec=int(copied / secs) if secs else 0, generator=f"sync from {vp}")
            WH_SYNC["result"] = {"tables": per, "rows": sum(p["rows"] for p in per), "dim": dim,
                                 "binary_copy": binary}
            WH_SYNC["phase"] = "done"
        except Exception as e:
            WH_SYNC["error"] = str(e)
            WH_SYNC["phase"] = "error"
        finally:
            fp_invalidate()
            WH_SYNC["running"] = False
            for c in (pgc, wc):
                if c:
                    try: c.close()
                    except Exception: pass

    @app.route("/api/wh/sync", methods=["POST"])
    def wh_sync():
        if not WH["enabled"]:
            return jsonify({"ok": False, "error": "WarehousePG is not enabled — set it up in Setup → Connection."}), 400
        err = wh_error()
        if err:
            return jsonify({"ok": False, "error": wh_down_message(err)}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        which = (request.get_json(silent=True) or {}).get("which", "all")
        if which not in ("all", "ivfplus", "hnsw", "ivfflat"):
            return jsonify({"ok": False, "error": "which must be all, ivfplus, hnsw or ivfflat"}), 400
        if not exists(vp_table()) or not count_rows(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} is missing or empty on Postgres — load it first."}), 400
        tbls = list(WH_TABLES.values()) if which == "all" else [WH_TABLES[which]]
        threading.Thread(target=_wh_sync_worker, args=(tbls,), daemon=True).start()
        return jsonify({"ok": True, "tables": tbls})

    @app.route("/api/wh/sync/status")
    def wh_sync_status():
        s = dict(WH_SYNC)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 2)
        s["pct"] = round(100 * s["done"] / s["total"], 1) if s["total"] else 0
        return jsonify(s)

    def pg_engine(base):
        return next(e for e in engines(include_wh=False) if e["key"] == base)

    def _wh_build_worker(sql, name, table, am, opclass, opts, maint_mem, to_drop):
        WH_BUILD.update({"running": True, "sql": sql, "started": time.time(), "elapsed": 0,
                         "error": None, "notices": [], "phase": "building", "result": None,
                         "name": name, "am": am, "maintenance_work_mem": maint_mem})
        conn = None
        try:
            rows = count_rows(table, "whpg")
            conn = wh_connect(autocommit=True)
            cur = conn.cursor()
            soft(cur, "SET statement_timeout = 0")
            for old in to_drop or []:
                cur.execute(f'DROP INDEX IF EXISTS "{schema("whpg")}"."{old}"')
                log_query("ddl", f'DROP INDEX IF EXISTS "{schema("whpg")}"."{old}"', label=old)
            if maint_mem:
                soft(cur, f"SET maintenance_work_mem = '{maint_mem}'")
            t0 = time.perf_counter()
            cur.execute(sql)
            elapsed = time.perf_counter() - t0
            log_query("index-build-whpg", sql, ms=elapsed * 1000, rows=rows, label=name,
                      settings=WH_SETTINGS_FOR_LOG())
            WH_BUILD["notices"] = clean_notices(conn.notices)
            WH_BUILD["elapsed"] = round(elapsed, 3)
            cur.execute("SELECT pg_relation_size(%s::regclass)", (f"{schema('whpg')}.{name}",))
            size_b = int(cur.fetchone()[0] or 0)
            rec = {"name": name, "am": am, "table": table, "opclass": opclass, "opts": opts,
                   "seconds": round(elapsed, 3), "rows": rows,
                   "rows_per_sec": int(rows / elapsed) if elapsed > 0 else 0,
                   "bytes": size_b, "size": human_bytes(size_b),
                   "bytes_per_row": round(size_b / rows, 1) if rows else 0,
                   "maintenance_work_mem": maint_mem, "notices": WH_BUILD["notices"], "sql": sql}
            rec["id"] = store_insert(
                "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target=WH_TARGET,
                label=f"{WH_LABEL[am]} @ WHPG", engine=am, am=am, dsn_db=WH["database"], rows=rows,
                dim=dim_of(table, "whpg"), lists=opts.get("lists"), m=opts.get("m"),
                ef_construction=opts.get("ef_construction"), opclass=opclass, concurrently=0,
                index_name=name, build_seconds=round(elapsed, 3),
                rows_per_sec=round(rows / elapsed, 1) if elapsed > 0 else 0, index_bytes=size_b,
                bytes_per_row=round(size_b / rows, 2) if rows else 0, maint_work_mem=maint_mem,
                parallel_workers=f"{WH['segments']} segments" if WH["segments"] else None,
                pg_version=pscalar("whpg", "SELECT version()"),
                ext_version=pscalar("whpg", "SELECT extversion FROM pg_extension WHERE extname = %s",
                                    ("edb_vectorplus" if am == "ivfplus" else "vector",)),
                notices=" | ".join(WH_BUILD["notices"]), note="; ".join(WH["session"]))
            WH_BUILD["result"] = rec
            WH_BUILD["phase"] = "done"
        except Exception as e:
            WH_BUILD["error"] = str(e)
            WH_BUILD["phase"] = "error"
            if conn:
                WH_BUILD["notices"] = clean_notices(conn.notices)
        finally:
            WH_BUILD["running"] = False
            if conn:
                try: conn.close()
                except Exception: pass

    def WH_SETTINGS_FOR_LOG():
        return [s.replace("SET ", "", 1) for s in wh_settings_sql()]

    @app.route("/api/wh/index/build/start", methods=["POST"])
    def wh_build_start():
        if not WH["enabled"]:
            return jsonify({"ok": False, "error": "WarehousePG is not enabled."}), 400
        err = wh_error()
        if err:
            return jsonify({"ok": False, "error": wh_down_message(err)}), 400
        if WH_BUILD["running"]:
            return jsonify({"ok": False, "error": "A WarehousePG build is already running"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        body = request.get_json(silent=True) or {}
        am = body.get("am", "hnsw")
        if am not in WH_TABLES:
            return jsonify({"ok": False, "error": "am must be ivfplus, hnsw or ivfflat"}), 400
        opclass = body.get("opclass", "vector_l2_ops")
        if opclass not in G["OPCLASSES"]:
            return jsonify({"ok": False, "error": f"Unknown opclass {opclass}"}), 400
        table = WH_TABLES[am]
        if not exists(table, "whpg"):
            return jsonify({"ok": False, "error": f"{table} does not exist on WarehousePG — sync it first."}), 400
        rows = count_rows(table, "whpg")
        if not rows:
            return jsonify({"ok": False, "error": f"{table} is empty — sync it first."}), 400
        dim = dim_of(table, "whpg")
        if not opclass.startswith("halfvec") and dim > 2000:
            return jsonify({"ok": False, "error": f"vector indexes stop at 2000 dimensions; this is {dim}. "
                            "Use a halfvec opclass."}), 400

        # Parameters default to whatever the Postgres index of the same method was built with,
        # untouched: the same lists on WarehousePG, not divided by the segment count.
        src = None
        try:
            idx = engine_indexes(pg_engine(am))
            src = idx[0] if idx else None
        except Exception:
            pass
        copied = {}

        def pick(field, given, cast=int):
            if given not in (None, ""):
                return cast(given)
            if src is not None and src.get(field) not in (None, []):
                v = src[field][0] if isinstance(src[field], list) else src[field]
                copied[field] = v
                return cast(v)
            raise ValueError(f"No {field} given and there is no Postgres {am} index to copy it from — "
                             "enter a value or build the Postgres index first.")
        try:
            if am == "hnsw":
                m_ = pick("m", body.get("m"))
                efc = pick("ef_construction", body.get("ef_construction"))
                if not (2 <= m_ <= 100) or not (4 <= efc <= 1000) or efc < 2 * m_:
                    raise ValueError("m must be 2–100 and ef_construction 4–1000 and at least 2 × m")
                opts, with_, tag = {"m": m_, "ef_construction": efc}, f"m = {m_}, ef_construction = {efc}", f"m{m_}_ef{efc}"
            else:
                lists = pick("lists", body.get("lists"))
                if not (1 <= lists <= 32768):
                    raise ValueError("lists must be 1–32768")
                opts, with_, tag = {"lists": lists}, f"lists = {lists}", f"l{lists}"
                if am == "ivfplus":
                    with_ += f", rotation = {'on' if body.get('rotation', True) else 'off'}"
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        maint = (body.get("maintenance_work_mem") or "").strip() or None
        if maint and not _MWM_RE.match(maint):
            return jsonify({"ok": False, "error": "maintenance_work_mem must look like 512MB or 2GB"}), 400
        existing = am_indexes(table, am, "whpg")
        to_drop = [i["name"] for i in existing] if body.get("replace", True) else []
        name = (body.get("name") or "").strip()
        if name:
            if not re.fullmatch(r"[A-Za-z0-9_]{1,60}", name):
                return jsonify({"ok": False, "error": "Index name must be letters, digits and underscores"}), 400
        else:
            short = opclass.replace("vector_", "v").replace("halfvec_", "h").replace("_ops", "")
            base, taken, n = f"{table}_emb_{am}_{tag}_{short}", {i["name"] for i in existing if i["name"] not in to_drop}, 2
            name = base
            while name in taken:
                name, n = f"{base}_{n}", n + 1
        col = f"(embedding::halfvec({dim}))" if opclass.startswith("halfvec") else "embedding"
        sql = (f"CREATE INDEX \"{name}\" ON {qual(table, 'whpg')}\n"
               f"    USING {am} ({col} {opclass})\n    WITH ({with_})")
        warnings = [f"Session settings applied to this connection first: {'; '.join(WH['session']) or 'none'}."]
        if copied:
            warnings.append("Parameters copied unchanged from the Postgres index: " +
                            ", ".join(f"{k} = {v}" for k, v in copied.items()) +
                            " — not scaled for the segment count.")
        if to_drop:
            warnings.append(f"Replacing existing index {', '.join(to_drop)} — dropped first and not timed.")
        if WH["segments"]:
            warnings.append(f"{WH['segments']} segments: each builds its own index over about "
                            f"{rows // max(1, WH['segments']):,} rows, so this is {WH['segments']} smaller "
                            "builds in parallel, not one large one.")
        threading.Thread(target=_wh_build_worker,
                         args=(sql, name, table, am, opclass, opts, maint, to_drop), daemon=True).start()
        return jsonify({"ok": True, "sql": sql, "name": name, "warnings": warnings, "copied": copied})

    @app.route("/api/wh/index/build/status")
    def wh_build_status():
        s = dict(WH_BUILD)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 2)
        return jsonify(s)

    @app.route("/api/wh/index/list")
    def wh_index_list():
        rows = []
        if WH["enabled"]:
            for am, tb in WH_TABLES.items():
                for r in am_indexes(tb, am, "whpg"):
                    r["am"], r["table"] = am, tb
                    rows.append(r)
        hist = store_query("SELECT * FROM builds WHERE target = ? ORDER BY id DESC LIMIT 50", (WH_TARGET,))
        for h in hist:
            h["at"] = (h.get("ts") or "")[5:16]
            h["size"] = human_bytes(h.get("index_bytes"))
        return jsonify({"ok": True, "rows": rows, "history": hist})

    @app.route("/api/wh/index/drop", methods=["POST"])
    def wh_index_drop():
        name = (request.get_json(silent=True) or {}).get("name") or ""
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            return jsonify({"ok": False, "error": "Bad index name"}), 400
        conn = None
        try:
            conn = wh_connect(autocommit=True)
            sql = f'DROP INDEX IF EXISTS "{schema("whpg")}"."{name}"'
            conn.cursor().execute(sql)
            log_query("ddl", sql, label=name)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        finally:
            if conn:
                try: conn.close()
                except Exception: pass

    # ════════════════════════════════════════════════════════════════════════
    #  LOADER — why a reload crawls, and where the time goes
    # ════════════════════════════════════════════════════════════════════════
    # COPY into a table that already carries a vector index inserts every row into that
    # index one at a time. Measured: ~18,000 rows/s into a plain table, ~400–600 rows/s
    # with an HNSW index present — and an ivfplus index only parks the rows in its slower
    # append region. Either way the index has to be rebuilt afterwards (clustering happens
    # at build time only), so the loader now drops vector indexes first unless told not to.

    VECTOR_AMS = ("ivfplus", "ivfflat", "hnsw", "vchordrq")

    def table_vector_indexes():
        return [r["name"] for am in VECTOR_AMS for r in am_indexes(vp_table(), am)]

    @app.before_request
    def _prepare_load():
        if request.method != "POST" or request.path != "/api/data/generate/start":
            return None
        g.dropped_vec, g.kept_vec = [], []
        if G["GEN_STATE"]["running"] or G["BUILD_STATE"]["running"]:
            return None                      # the handler refuses; touch nothing
        body = request.get_json(silent=True) or {}
        names = table_vector_indexes()
        if names and body.get("drop_vector_indexes", True):
            conn = None
            try:
                conn = get_conn(autocommit=True)
                cur = conn.cursor()
                for n in names:
                    sql = f'DROP INDEX IF EXISTS "{schema()}"."{n}"'
                    cur.execute(sql)
                    log_query("ddl", sql, label=n)
                    g.dropped_vec.append(n)
            except Exception as e:
                print(f"  [loader] could not drop vector indexes: {e}")
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
        else:
            g.kept_vec = names
        st = G["GEN_STATE"]
        st["gen_seconds"], st["copy_live_seconds"] = 0.0, 0.0
        return None

    @app.after_request
    def _fp_hooks(resp):
        """Loads, resets and table creation change rows outside any worker of ours."""
        try:
            if request.method == "POST" and request.path in ("/api/data/generate/start", "/api/data/generate/stop",
                                                              "/api/setup/reset", "/api/setup/create_table"):
                fp_invalidate()
            elif request.path == "/api/data/generate/status" and resp.is_json:
                d = resp.get_json(silent=True)
                if isinstance(d, dict) and d.get("phase") in ("done", "error"):
                    fp_invalidate()
        except Exception:
            pass
        return resp

    @app.after_request
    def _report_load(resp):
        if request.method != "POST" or request.path != "/api/data/generate/start" or not resp.is_json:
            return resp
        try:
            data = resp.get_json()
            if not isinstance(data, dict) or not data.get("ok"):
                return resp
            warns = list(data.get("warnings") or [])
            dropped, kept = getattr(g, "dropped_vec", []), getattr(g, "kept_vec", [])
            if dropped:
                warns.insert(0, "Dropped before loading: " + ", ".join(dropped) + ". COPY into a table that "
                             "carries a vector index runs ~40× slower (every row is inserted into the index "
                             "one at a time) and the index has to be rebuilt after the load anyway — rebuild "
                             "it on the Index Build tab.")
            if kept:
                warns.insert(0, "Loading with " + ", ".join(kept) + " still in place, as asked. Expect a few "
                             "hundred rows/s rather than several thousand, and a rebuild afterwards.")
            data["warnings"], data["dropped_vector_indexes"] = warns, dropped
            resp.set_data(json.dumps(data))
        except Exception:
            pass
        return resp

    # time spent generating vs inside COPY, so a slow load says why instead of just being slow
    _orig_make_batch, _orig_copy_rows = G.get("make_batch"), G.get("copy_rows")

    def _make_batch_timed(*a, **k):
        t0 = time.perf_counter()
        out = _orig_make_batch(*a, **k)
        st = G["GEN_STATE"]
        if st.get("running") and st.get("phase") == "generating":
            st["gen_seconds"] = round(st.get("gen_seconds", 0.0) + time.perf_counter() - t0, 3)
        return out

    def _copy_rows_timed(*a, **k):
        t0 = time.perf_counter()
        out = _orig_copy_rows(*a, **k)
        st = G["GEN_STATE"]
        if st.get("running") and st.get("phase") == "generating":
            st["copy_live_seconds"] = round(st.get("copy_live_seconds", 0.0) + time.perf_counter() - t0, 3)
        return out

    if _orig_make_batch and _orig_copy_rows:        # the host app has both; a stub may not
        G["make_batch"], G["copy_rows"] = _make_batch_timed, _copy_rows_timed

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
        """Warm the index — not the heap. A query touches a handful of heap pages but walks the index,
        and loading a whole table that is larger than shared_buffers only evicts what was just read."""
        done = []
        try:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_prewarm'")
            have_pw = bool(cur.fetchone())
            cur.execute("SELECT pg_size_bytes(current_setting('shared_buffers'))")
            sb = int(cur.fetchone()[0])
        except Exception:
            have_pw, sb = False, 128 * 1024 * 1024
        for i in engine_indexes(e):
            try:
                if e["key"] == "vchordrq":
                    cur.execute("SELECT vchordrq_prewarm(%s::regclass)", (f"{schema()}.{i['name']}",))
                    done.append("vchordrq_prewarm")
                elif have_pw:
                    # into shared_buffers if it fits comfortably, otherwise into the OS cache only
                    mode = "buffer" if (i.get("bytes") or 0) <= sb * 0.5 else "read"
                    cur.execute("SELECT pg_prewarm(%s::regclass, %s)", (f"{schema()}.{i['name']}", mode))
                    done.append(f"pg_prewarm(index, {mode})")
            except Exception as ex:
                done.append("index: " + str(ex).strip().splitlines()[0])
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
            es = [e for e in engines(include_wh=False) if exists(e["table"]) and engine_indexes(e)
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
                lists = engine_indexes(e)[0].get("lists") or []
                if e["key"] == "vchordrq":
                    opts = default_vc_probes(lists)
                    p = opts[min(2, len(opts) - 1)]
                elif e["key"] == "ivfflat":
                    opts = default_ivf_probes(lists)
                    p = opts[min(2, len(opts) - 1)]
                elif e["key"] == "hnsw":
                    p = 40
                else:
                    p = 16
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
        key = key[2:] if key.startswith("w_") else key      # WarehousePG twins share their setting
        if key == "clickhouse":
            return [f"SET hnsw_candidate_list_size_for_search = {max(1, min(100000, int(p)))}"]
        if key == "clickhouse_qbit":
            return [f"SET qbit_bits = {max(1, min(64, int(p)))}"]
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

    def ann_sql(table, opclass, limit, category_id, prof="pg"):
        if prof == "ch":
            return ann_sql_ch(table, opclass, limit, category_id)
        meta = G["OPCLASSES"][opclass]
        op, col, cast = meta["op"], "embedding", ""
        if opclass.startswith("halfvec"):
            d = dim_of(table, prof)
            col, cast = f"embedding::halfvec({d})", f"::halfvec({d})"
        where = f"WHERE category_id = {int(category_id)}\n" if category_id not in (None, "", "any") else ""
        return (f"SELECT id, topic, category_id, title,\n"
                f"       {col} {op} %s{cast} AS distance\n"
                f"FROM {qual(table, prof)}\n{where}"
                f"ORDER BY {col} {op} %s{cast}\nLIMIT {int(limit)}")

    def plan_of(cur, sql, params):
        """Which index, if any, the planner actually chose — recall of 1.0 at high
        latency is the signature of a quiet sequential scan."""
        if isinstance(cur, ChCursor):
            return chd_plan(cur, sql, params)
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
        soft(cur, f"SET vchordrq.enable_scan = {'off' if exact else 'on'}")
        if exact:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
            cur.execute("SET enable_seqscan = on")
        else:
            cur.execute("SET enable_indexscan = on")
            cur.execute("SET enable_bitmapscan = on")
            cur.execute(f"SET enable_seqscan = {'off' if force_index else 'on'}")

    def run_ann(cur, table, qv, opclass, limit, category_id, setters,
                exact=False, force_index=True, reps=1, warm=True, kind="cmp", clean=None, prof="pg", tick=None):
        """One query, run `reps` times. In cold mode (`clean` given) the caches are
        evicted right before the first run, so that run is the cold figure and the
        rest are the warm ones; otherwise a throwaway run precedes the timed ones."""
        sql = ann_sql(table, opclass, limit, category_id, prof)
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
            if tick: tick("warm-up run")
        times, rows = [], []
        for i_ in range(max(1, reps)):
            t0 = time.perf_counter()
            cur.execute(sql, params)
            rows = cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
            if tick: tick("scan" if exact else f"timed run {i_ + 1} of {max(1, reps)}")
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
    def engines(include_wh=True):
        """Every engine this dashboard can compare, each on its own table of identical
        rows, each bound to a connection profile ('pg' or 'whpg'). A table that does
        not exist simply drops out."""
        out = [
            {"key": "ivfplus", "base": "ivfplus", "prof": "pg", "table": vp_table(), "am": "ivfplus",
             "target": G["TARGET_NAME"], "label": G["TARGET_LABEL"], "param": "ivfplus.probes"},
            {"key": "vchordrq", "base": "vchordrq", "prof": "pg", "table": VC_TABLE, "am": "vchordrq",
             "target": VC_TARGET, "label": VC_LABEL, "param": "vchordrq.probes"},
            {"key": "hnsw", "base": "hnsw", "prof": "pg", "table": PGH_TABLE, "am": "hnsw",
             "target": PGV_TARGET, "label": "pgvector HNSW", "param": "hnsw.ef_search"},
            {"key": "ivfflat", "base": "ivfflat", "prof": "pg", "table": PGI_TABLE, "am": "ivfflat",
             "target": PGV_TARGET, "label": "pgvector IVFFlat", "param": "ivfflat.probes"}]
        if include_wh and CH["enabled"]:
            out.append({"key": "clickhouse", "base": "clickhouse", "prof": "ch", "table": CH_TABLE, "am": "clickhouse",
                        "target": CH_TARGET, "label": "ClickHouse", "param": "hnsw_candidate_list_size_for_search"})
            out.append({"key": "clickhouse_qbit", "base": "clickhouse_qbit", "prof": "ch", "table": CHQ_TABLE, "am": "clickhouse_qbit",
                        "target": CH_TARGET, "label": "ClickHouse QBit", "param": "bits read per number"})
        if include_wh and WH["enabled"]:
            for base, label, param in (("ivfplus", "EDB ivfplus", "ivfplus.probes"),
                                       ("hnsw", "pgvector HNSW", "hnsw.ef_search"),
                                       ("ivfflat", "pgvector IVFFlat", "ivfflat.probes")):
                out.append({"key": "w_" + base, "base": base, "prof": "whpg", "table": WH_TABLES[base],
                            "am": base, "target": WH_TARGET, "label": f"{label} @ WHPG", "param": param})
        return out

    def engines_by_key():
        return {e["key"]: e for e in engines()}

    def engine_indexes(e):
        prof = e.get("prof", "pg")
        if prof == "ch":
            return chq_indexes() if e.get("key") == "clickhouse_qbit" else chd_indexes()
        if e["key"] == "ivfplus":                 # Postgres ivfplus: the host app already knows these
            out = []
            for r in G["ivfplus_indexes"]():
                if r["am"] != "ivfplus":
                    continue
                lists = r.get("lists")            # the host app parses this as a bare int
                if isinstance(lists, int):
                    lists = [lists]
                out.append(dict(r, lists=lists, size=human_bytes(r["bytes"])))
            return out
        return am_indexes(e["table"], e["am"], prof)

    def where_of(e):
        return {"whpg": " on WarehousePG", "ch": " in ClickHouse"}.get(e.get("prof"), "")

    VERIFY = {}

    def need_engines(keys=None, need_index=True, job=None):
        """The engines to compare, validated: at least two, identical row sets everywhere
        (including across clusters), and exactly one index each so the planner's choice is
        never in question."""
        if not keys or any(k.startswith("w_") for k in keys):
            err = wh_error()
            if err:
                raise ValueError(wh_down_message(err))
        if CH["enabled"] and (not keys or "clickhouse" in keys):
            err = ch_error()
            if err:
                raise ValueError(f"ClickHouse is enabled but cannot be reached: {err}. Start the service and press "
                                 "Connect on the Setup page, or untick ClickHouse.")
        es = [e for e in engines() if exists(e["table"], e["prof"]) and (not keys or e["key"] in keys)]
        if keys:
            miss = [k for k in keys if k not in {e["key"] for e in es}]
            if miss:
                raise ValueError("No table yet for: " + ", ".join(miss) +
                                 " — clone or sync it from its tab first")
        if len(es) < 2:
            raise ValueError("Need at least two engine tables to compare — clone vc_docs / pgh_docs / "
                             "pgi_docs, or sync the WarehousePG tables, from their tabs")
        base = es[0]
        # One full scan per table, all at once. It is the slow part of "preparing", so say which are done
        # and which we are still waiting on — and keep the timings so a slow table can be named afterwards.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        t_start, pending, took = time.time(), {e_["key"]: e_["label"] for e_ in es}, {}
        if job:
            job["phase"] = "verifying"
            job["current"] = f"checking that {len(es)} tables hold identical rows (a full scan of each)…"
        def _one(e_):
            t0_ = time.time()
            return e_["key"], fingerprint(e_["table"], e_["prof"]), round(time.time() - t0_, 1)
        fp_by = {}
        with ThreadPoolExecutor(max_workers=len(es)) as ex:
            for fut in as_completed([ex.submit(_one, e_) for e_ in es]):
                k_, fp_, sec_ = fut.result()
                fp_by[k_], took[k_] = fp_, sec_
                pending.pop(k_, None)
                if job:
                    job["current"] = (f"checking that the tables hold identical rows — {len(es) - len(pending)} of {len(es)} done"
                                      + (f"; still scanning: {', '.join(pending.values())}" if pending else ""))
        fps = [fp_by[e_["key"]] for e_ in es]
        slow = max(took, key=took.get)
        VERIFY["last"] = {"seconds": round(time.time() - t_start, 1), "slowest": next(e_["label"] for e_ in es if e_["key"] == slow),
                          "slowest_s": took[slow], "took": took}
        d0, fp0 = dim_of(base["table"], base["prof"]), fps[0]
        for e, fp in zip(es[1:], fps[1:]):
            if dim_of(e["table"], e["prof"]) != d0:
                raise ValueError(f"{e['table']}{where_of(e)} is {dim_of(e['table'], e['prof'])}-dimensional "
                                 f"but {base['table']} is {d0} — re-clone")
            if fp != fp0:
                raise ValueError(f"{e['table']}{where_of(e)} does not hold the same rows as {base['table']} "
                                 f"({fp[0]:,} rows, ids {fp[1]}–{fp[2]} vs {fp0[0]:,} rows, ids {fp0[1]}–{fp0[2]}"
                                 f"{', same range but different id checksum' if fp[:3] == fp0[:3] else ''}) — "
                                 "re-clone or re-sync so every table holds identical rows")
        if need_index:
            for e in es:
                idx = engine_indexes(e)
                if not idx:
                    raise ValueError(f"{e['table']}{where_of(e)} has no {e['am']} index — build one first "
                                     f"(or untick {e['label']})")
                if len(idx) > 1 and e["key"] != "ivfplus":
                    raise ValueError(f"{e['table']}{where_of(e)} has {len(idx)} {e['am']} indexes "
                                     f"({', '.join(i['name'] for i in idx)}). The planner chooses between "
                                     "them, which makes the comparison ambiguous — drop all but one.")
        return es

    def truth_engine(es):
        """Exact scans are identical on every table; run them on Postgres, on a table that
        is not vchordrq's (its planner insists on probes), and only elsewhere if nothing else exists."""
        pg = [e for e in es if e["prof"] == "pg"]
        return next((e for e in pg if e["key"] != "vchordrq"), (pg or es)[0])

    def vp_index_note():
        idx = [r for r in G["ivfplus_indexes"]() if r["am"] == "ivfplus"]
        if len(idx) > 1:
            return (f"{vp_table()} has {len(idx)} ivfplus indexes; the planner chose between them "
                    f"({', '.join(i['name'] for i in idx)}). Drop the others for a clean comparison.")
        return None

    def _pmap(body):
        """Per-engine setting. WarehousePG engines use the same value as their Postgres twin,
        so one box drives both and the comparison stays like for like."""
        base = {"ivfplus": body.get("vp_probes", 10), "vchordrq": body.get("vc_probes", ""),
                "hnsw": body.get("hnsw_ef", 40), "ivfflat": body.get("ivf_probes", 10),
                "clickhouse": body.get("hnsw_ef", 40), "clickhouse_qbit": body.get("qbit_bits", 16)}
        base.update({"w_" + k: v for k, v in list(base.items()) if k != "vchordrq"})
        return base

    HNSW_EF_PER_PROBE = 4

    def effort_setting(e, p, k, lists, ef_per=HNSW_EF_PER_PROBE):
        """The engine's own setting for one shared 'probes' value. IVF-style engines take it as probes
        (never above their lists); VectorChord likewise, with its second level at the usual default;
        HNSW has no probes, so it gets ef_search = max(k, ef_per x probes)."""
        p, base = int(p), e["base"]
        if base == "clickhouse_qbit":
            return max(1, min(32, p))                     # bits of each Float32 that are read: 1 to 32
        if base in ("ivfplus", "ivfflat"):
            return min(p, lists[0]) if lists else p
        if base == "vchordrq":
            if not lists:
                return ""
            first = min(p, lists[0])
            return str(first) if len(lists) == 1 else f"{first},{max(1, lists[1] // 100)}"
        return max(1, min(1000, max(int(k), ef_per * p)))

    def whpg_note():
        seg = WH.get("segments")
        return ("WarehousePG runs each query on every segment"
                + (f" ({seg} here)" if seg else "") + " and merges at the coordinator: probes / ef_search "
                "apply per segment, so the same setting does more total work than on Postgres, and latency "
                "includes coordinator overhead. Compare at equal recall, not at equal settings. "
                f"Session settings in force: {'; '.join(WH['session']) or 'none'}.")

    # ── summary ─────────────────────────────────────────────────────────────
    @app.route("/api/cmp/summary")
    def cmp_summary():
        werr, cerr = wh_error(), ch_error()
        out = {"ok": True, "engines": {}, "order": [], "whpg_enabled": WH["enabled"],
               "segments": WH.get("segments"), "whpg_error": werr, "ch_enabled": CH["enabled"], "ch_error": cerr}
        for e in engines():
            ex = ((not werr or e["prof"] != "whpg") and (not cerr or e["prof"] != "ch")) and exists(e["table"], e["prof"])
            card = {"table": e["table"], "label": e["label"], "engine": e["am"], "key": e["key"],
                    "prof": e["prof"], "exists": ex}
            if ex:
                _fp = fingerprint_cached(e["table"], e["prof"])
                card["rows"], card["rows_exact"] = (_fp[0] if _fp else rows_est(e["table"], e["prof"])), bool(_fp)
                card["dim"] = dim_of(e["table"], e["prof"])
                if e["prof"] == "ch":
                    try:
                        card["heap_bytes"] = int(ch_rows(f"SELECT sum(data_compressed_bytes) AS b FROM system.parts WHERE database = '{CH['database']}' "
                                                         f"AND table = '{e['table']}' AND active")[0]["b"] or 0)
                    except Exception:
                        card["heap_bytes"] = 0
                else:
                    card["heap_bytes"] = int(pscalar(e["prof"], "SELECT pg_table_size(%s::regclass)",
                                                     (f"{schema(e['prof'])}.{e['table']}",), default=0) or 0)
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

    def refuse_cold_on_whpg(es):
        bad = [e["label"] for e in es if e["prof"] != "pg"]
        if bad:
            raise ValueError("Cold runs clear caches on the Postgres host only — the caches of WarehousePG's segment "
                             "hosts and of ClickHouse are out of reach. Untick " + ", ".join(bad) + ", or use warm mode.")

    # ── one query, every engine ─────────────────────────────────────────────
    def _search_core(body, job=None):
        CN = None
        try:
            es = need_engines(body.get("engines"), job=job)
            qv, label = G["resolve_query_vector"](body)
            opclass = body.get("opclass", "vector_l2_ops")
            if opclass not in G["OPCLASSES"]:
                raise ValueError("Unknown opclass")
            k = max(1, min(200, int(body.get("limit", 10))))
            cat = body.get("category_id")
            reps = max(1, min(20, int(body.get("reps", 3))))
            force = bool(body.get("force_index", True))
            pm = _pmap(body)
            if body.get("probes") not in (None, ""):
                for e_ in es:
                    ix_ = engine_indexes(e_)
                    pm[e_["key"]] = effort_setting(e_, body["probes"], k, (ix_[0].get("lists") if ix_ else None),
                                                   int(body.get("hnsw_ef_per_probe") or HNSW_EF_PER_PROBE))
            pre = body.get("prefilter")
            pre = None if pre is None else bool(pre)
            it = body.get("iterative") or "off"
            cold = body.get("cache_mode") == "cold"

            CN = Conns(120000, registry=(job.setdefault("pids", {}) if job else None))
            if cold:
                refuse_cold_on_whpg(es)
                caps = cache_caps(CN.cursor("pg"))
                require_cold(caps)
                reps = max(reps, 3)           # first run is cold, the rest give the warm figure
            def mk_tick(label):
                """One call per statement executed, so the page can say 'N of M queries'."""
                if not job:
                    return None
                def tick(stage=""):
                    if job.get("cancel"):
                        raise RuntimeError("Search cancelled.")
                    job["done"] += 1
                    job["current"] = label + (f" · {stage}" if stage else "")
                return tick
            if job:
                job.update(total=1 + len(es) * (reps + (0 if cold else 1)), phase="running", current="starting")
            te = truth_engine(es)
            exact = run_ann(CN.cursor(te["prof"]), te["table"], qv, opclass, k, cat, [], exact=True,
                            kind="cmp-exact", prof=te["prof"], tick=mk_tick("exact scan (ground truth)"))
            truth = {r["id"] for r in exact["rows"]}

            outs = []
            for e in es:
                if e["key"] == "vchordrq":
                    check_vc_probes(engine_indexes(e)[0].get("lists"), pm["vchordrq"])
                cur = CN.cursor(e["prof"])
                setters = engine_setters(e["key"], pm[e["key"]], body.get("epsilon"), pre, it)
                res = run_ann(cur, e["table"], qv, opclass, k, cat, setters, force_index=force, reps=reps,
                              kind="cmp-" + e["key"], prof=e["prof"], tick=mk_tick(e["label"]),
                              clean=(lambda t=e["table"], c=cur: clean_caches(c, t)) if cold else None)
                ids = {r["id"] for r in res["rows"]}
                outs.append({"key": e["key"], "label": e["label"], "table": e["table"], "prof": e["prof"],
                             "result": res, "settings": setters, "ids": sorted(ids),
                             "recall": round(len(truth & ids) / len(truth), 4) if truth else None,
                             "returned": len(res["rows"]),
                             "speedup": round(exact["ms"] / res["ms"], 1) if res["ms"] > 0 else None})
            common = set(outs[0]["ids"])
            for o in outs[1:]:
                common &= set(o["ids"])

            notes = [n for n in (vp_index_note(),) if n]
            if any(e["prof"] == "whpg" for e in es):
                notes.append(whpg_note())
            if any(e["key"] == "clickhouse" for e in es):
                w_ = chd_cache_info()["warning"]
                if w_:
                    notes.append("ClickHouse: " + w_)
            if any(e["key"] == "clickhouse_qbit" for e in es):
                notes.append("ClickHouse QBit has no index: every query scans every row, reading only the top bits of each number. "
                             "The setting is bits read (1 to 32), so latency grows with the table size, and the lowest settings "
                             "read too little of each number to rank neighbours.")
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
                                 f"sequential scan — build an index with opclass {opclass}"
                                 + (" (on WarehousePG this often means the optimizer setting did not apply)"
                                    if o["prof"] == "whpg" else "") + ".")
                if o["key"].endswith("hnsw") and it == "off" and k > int(pm[o["key"]]):
                    notes.append(f"{o['label']}: LIMIT {k} is above hnsw.ef_search ({pm[o['key']]}), so it "
                                 f"can return at most {pm[o['key']]} rows per segment. Raise ef_search, or "
                                 "turn on iterative scan.")
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
            return ({"ok": True, "label": label, "k": k, "opclass": opclass,
                            "mapped": {e_["key"]: pm[e_["key"]] for e_ in es},
                            "reps": reps, "force_index": force, "exact": exact,
                            "engines": outs, "common_to_all": sorted(common), "notes": notes,
                            "protocol": {"cache_mode": "cold" if cold else "warm", "reps": reps, "jit": "off"}})
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            if CN:
                CN.close()


    @app.route("/api/cmp/search", methods=["POST"])
    def cmp_search():
        """Synchronous form, kept for scripts. The dashboard uses /start and polls."""
        res = _search_core(request.get_json(silent=True) or {})
        return jsonify(res), (200 if res.get("ok") else 400)

    def _search_thread(body):
        try:
            res = _search_core(body, SEARCH)
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        if SEARCH.get("cancel") and not res.get("ok"):
            res = {"ok": False, "error": "Search cancelled."}
        SEARCH["result"] = res if res.get("ok") else None
        SEARCH["error"] = None if res.get("ok") else res.get("error")
        SEARCH["phase"] = "done" if res.get("ok") else ("cancelled" if "cancelled" in str(res.get("error")) else "error")
        SEARCH["elapsed"] = round(time.time() - SEARCH["started"], 1)
        SEARCH["running"] = False

    @app.route("/api/cmp/search/start", methods=["POST"])
    def cmp_search_start():
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is already running — wait for it."}), 400
        SEARCH.update(running=True, phase="preparing", done=0, total=0, current="", started=time.time(), elapsed=0,
                      error=None, result=None, cancel=False, pids={})
        threading.Thread(target=_search_thread, args=(request.get_json(silent=True) or {},), daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/cmp/search/status")
    def cmp_search_status():
        s_ = dict(SEARCH)
        s_.pop("pids", None)
        if s_["running"] and s_["started"]:
            s_["elapsed"] = round(time.time() - s_["started"], 1)
        s_["pct"] = round(100 * s_["done"] / s_["total"], 1) if s_["total"] else 0
        return jsonify(s_)

    @app.route("/api/cmp/search/cancel", methods=["POST"])
    def cmp_search_cancel():
        SEARCH["cancel"] = True
        for prof, pid in list((SEARCH.get("pids") or {}).items()):
            conn = None
            try:
                conn = prof_conn(prof, autocommit=True)
                conn.cursor().execute("SELECT pg_cancel_backend(%s)", (int(pid),))
            except Exception:
                pass
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
        return jsonify({"ok": True})

    # ── sweep ───────────────────────────────────────────────────────────────
    def sample_queries(cur, table, nq, seed=7, prof="pg"):
        cur.execute(f"SELECT min(id), max(id) FROM {qual(table, prof)}")
        lo, hi = cur.fetchone()
        rng = np.random.default_rng(seed)
        ids = sorted({int(x) for x in rng.integers(int(lo), int(hi) + 1, size=nq * 3)})
        cur.execute(f"SELECT embedding::text FROM {qual(table, prof)} WHERE id = ANY(%s) LIMIT %s",
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

    def _recall_core(body, job=None):
        """
        The measurement protocol, since latency at this scale is dominated by cache state and
        ordering rather than by the index:

          * every engine's setting i is measured before any engine's setting i+1, and the
            engine order rotates, so no engine is always first or always benefits from another's reads;
          * each setting is measured over `passes` full passes; a query's latency is the median
            of its passes, which discards one-off stalls;
          * warm mode prewarms the Postgres indexes and does a throwaway pass per setting; cold
            mode evicts the engine's table and index from every reachable cache layer before EVERY pass;
          * recall carries a standard error, latency a pass-to-pass deviation, so two numbers that
            differ by less than their noise are visibly a tie.

        Engines on WarehousePG run on their own connection (with that profile's session
        settings) against the same query vectors and the same exact ground truth.
        """
        CN = None
        try:
            es = need_engines(body.get("engines"), job=job)
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
            if cold:
                refuse_cold_on_whpg(es)

            idx_of = {e["key"]: engine_indexes(e) for e in es}
            lists_of = lambda key: (idx_of[key][0].get("lists") if idx_of.get(key) else None)
            given = {"ivfplus": body.get("vp_probes_list"), "vchordrq": body.get("vc_probes_list"),
                     "hnsw": body.get("hnsw_ef_list"), "ivfflat": body.get("ivf_probes_list")}
            sweeps = {}
            pl = sorted({int(x) for x in (body.get("probes_list") or []) if str(x).strip().isdigit() and int(x) > 0})
            ef_per = int(body.get("hnsw_ef_per_probe") or HNSW_EF_PER_PROBE)
            for e in es:
                g, key, base = given.get(e["base"]), e["key"], e["base"]
                if pl:                       # the single list the dashboard sends
                    vals = []
                    for p_ in pl:
                        v_ = effort_setting(e, p_, k, lists_of(key), ef_per)
                        if v_ not in vals:
                            vals.append(v_)
                    sweeps[key] = vals
                    if base == "vchordrq":
                        for p_ in vals:
                            check_vc_probes(lists_of(key), p_)
                    continue
                if base == "ivfplus":
                    sweeps[key] = sorted({max(1, min(32768, int(p))) for p in
                                          (g or [1, 2, 4, 8, 16, 32, 64, 128])})
                elif base == "vchordrq":
                    sweeps[key] = [probes_ok(p) for p in (g or default_vc_probes(lists_of(key)))]
                    for p in sweeps[key]:
                        check_vc_probes(lists_of(key), p)
                elif base == "clickhouse_qbit":
                    sweeps[key] = sorted({max(1, min(32, int(p))) for p in (g or [1, 2, 4, 8, 12, 16, 20, 24, 32])})
                elif base in ("hnsw", "clickhouse"):
                    sweeps[key] = sorted({max(1, min(1000, int(p))) for p in (g or default_ef_list(k))})
                else:
                    sweeps[key] = sorted({max(1, int(p)) for p in (g or default_ivf_probes(lists_of(key)))})

            CN = Conns(0, registry=(job.setdefault("pids", {}) if job else None))
            if job: job["phase"] = "ground truth"
            caps = None
            if cold:
                caps = cache_caps(CN.cursor("pg"))
                require_cold(caps)
            te = truth_engine(es)
            tcur = CN.cursor(te["prof"])
            qvs = sample_queries(tcur, te["table"], nq, prof=te["prof"])
            if not qvs:
                raise ValueError("Could not sample query vectors")
            nq = len(qvs)
            lits = [vec_literal(v) for v in qvs]

            def setters_for(e, p):
                return engine_setters(e["key"], p, eps if e["base"] == "vchordrq" else None,
                                      None, it if e["base"] in ("hnsw", "ivfflat") else None)

            def run_pass(e, qsql, setters, subset=None):
                cur = CN.cursor(e["prof"])
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

            # Ground truth: one exact scan per query vector. It depends only on the rows, the query vectors,
            # k and the operator class — not on any index or setting — so it is computed once and kept.
            # (It is stored the moment it is complete, so a cancelled or crashed sweep still keeps it.)
            import hashlib
            tsql = ann_sql(te["table"], opclass, k, None, te["prof"])
            fp_t = fingerprint(te["table"], te["prof"])
            qhash = hashlib.sha1(np.ascontiguousarray(np.stack(qvs)).tobytes()).hexdigest()
            tkey = hashlib.sha1(json.dumps([fp_t, dim_of(te["table"], te["prof"]), opclass, k, qhash]).encode()).hexdigest()
            cached = None
            if not body.get("refresh_truth"):
                cached = bench_db(lambda c: (lambda r: dict(r) if r else None)(
                    c.execute("SELECT ts, exact_ms, truth FROM bench_truth WHERE key = ?", (tkey,)).fetchone()))
            if cached:
                truth = [set(x) for x in json.loads(cached["truth"])]
                exact_ms = json.loads(cached["exact_ms"])
                truth_note = f"exact baseline reused from {cached['ts']}"
                if job: job["current"] = "exact baseline reused from an earlier sweep"
            else:
                set_scan_mode(tcur, True, force)
                soft(tcur, f"SET max_parallel_workers_per_gather = {min(8, max(2, (os.cpu_count() or 4)))}")
                truth, exact_ms = [], []
                for qi, lit in enumerate(lits):
                    if job:
                        if job.get("cancel"):
                            raise RuntimeError("Sweep cancelled — nothing was saved.")
                        job["current"] = f"exact baseline (seq scan) · query {qi + 1} of {nq}"
                    t0 = time.perf_counter()
                    tcur.execute(tsql, (lit, lit))
                    truth.append({r[0] for r in tcur.fetchall()})
                    exact_ms.append((time.perf_counter() - t0) * 1000.0)
                truth_note = "exact baseline computed now"
                bench_db(lambda c: c.execute(
                    "INSERT OR REPLACE INTO bench_truth (key, ts, k, nq, rows, exact_ms, truth) VALUES (?,?,?,?,?,?,?)",
                    (tkey, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), k, nq, fp_t[0],
                     json.dumps(exact_ms), json.dumps([sorted(t_) for t_ in truth]))))
            exact_avg = round(float(np.median(exact_ms)), 2)

            qsqls = {e["key"]: ann_sql(e["table"], opclass, k, None, e["prof"]) for e in es}
            plans = {}
            for e in es:
                cur = CN.cursor(e["prof"])
                set_scan_mode(cur, False, force)
                for s in setters_for(e, sweeps[e["key"]][0]):
                    cur.execute(s)
                plans[e["key"]] = plan_of(cur, qsqls[e["key"]], (lits[0], lits[0]))

            prewarmed = {}
            if not cold and body.get("prewarm", True):
                if job: job["phase"] = "prewarming"
                for n_i, e in enumerate(es):
                    if job:
                        if job.get("cancel"):
                            raise RuntimeError("Sweep cancelled — nothing was saved.")
                        job["current"] = f"prewarming {e['label']} ({n_i + 1} of {len(es)})"
                    prewarmed[e["key"]] = (prewarm(CN.cursor("pg"), e) if e["prof"] == "pg"
                                           else ["none — throwaway pass only (not on Postgres)"])

            # interleave: setting i of every engine before setting i+1 of any, rotating who goes first
            schedule = []
            for i in range(max(len(v) for v in sweeps.values())):
                row = [e for e in es if i < len(sweeps[e["key"]])]
                r = i % len(row)
                schedule += [(e, i) for e in row[r:] + row[:r]]

            if job: job.update(total=len(schedule) * passes, phase="sweeping")
            results = {e["key"]: [None] * len(sweeps[e["key"]]) for e in es}
            cleans = {}
            for e, i in schedule:
                key, p = e["key"], sweeps[e["key"]][i]
                if job: job["current"] = f"{e['label']} · {e['param'].split('.')[-1]} = {p if p != '' else chr(39) * 2}"
                setters, qsql = setters_for(e, p), qsqls[key]
                if not cold:
                    run_pass(e, qsql, setters, subset=min(3, nq))      # throwaway, untimed
                grid, rec = [], None
                for pi in range(passes):
                    if job and job.get("cancel"):
                        raise RuntimeError("Sweep cancelled — nothing was saved.")
                    if cold:
                        cs = clean_caches(CN.cursor("pg"), e["table"])
                        cleans.setdefault(key, cs)
                    times, got = run_pass(e, qsql, setters)
                    grid.append(times)
                    if job: job["done"] += 1
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
            rows_n, dim = fingerprint(te["table"], te["prof"])[0], dim_of(te["table"], te["prof"])
            for e in es:
                bq = store_query("SELECT id FROM builds WHERE target = ? AND engine = ? "
                                 "ORDER BY id DESC LIMIT 1", (e["target"], e["am"]))
                bid = bq[0]["id"] if bq else None
                for r in results[e["key"]]:
                    first = str(r["param"]).split(",")[0]
                    extra = (f"probes='{r['param']}' epsilon={eps}; " if e["base"] == "vchordrq" else "") + \
                            (f"iterative_scan={it}; " if it != "off" and e["base"] in ("hnsw", "ivfflat") else "") + \
                            (f"session: {'; '.join(WH['session'])}; " if e["prof"] == "whpg" else "")
                    store_insert("recalls", ts=stamp, build_id=bid, target=e["target"],
                                 label=e["label"], engine=e["am"], index_name=plans[e["key"]]["index"],
                                 rows=rows_n, dim=dim, k=k, queries=nq, param=e["param"],
                                 value=int(first) if first.isdigit() else 0, recall=r["recall"],
                                 ms=r["ms"], p95_ms=r["p95_ms"], avg_returned=r["avg_returned"],
                                 exact_ms=exact_avg, speedup=r["speedup"], note=extra + proto)
            log_query("cmp-recall-sweep", tsql, settings=[f"k = {k}", f"{nq} query vectors", proto] +
                      [f"{e['key']}: {e['param']} over {sweeps[e['key']]}" for e in es],
                      ms=exact_avg, rows=sum(len(v) for v in results.values()) * nq * passes,
                      label=f"{opclass} · exact baseline {exact_avg}ms")

            notes = [n for n in (vp_index_note(),) if n]
            if not any(e["key"] == "clickhouse" for e in es) and not CH.get("quiet") and ch_local(CH):
                if ch_service_state(CH).get("state") == "active":
                    notes.append("ClickHouse is running on this host while these engines were measured. It shares memory and the "
                                 "page cache with Postgres, which can add latency spikes. For a quiet measurement tick 'Stop "
                                 "ClickHouse during the sweep' (or stop it on the Setup page).")
            v_ = VERIFY.get("last")
            if v_ and v_["seconds"] > 20:
                notes.append(f"Checking that the tables are identical took {v_['seconds']} s (slowest: {v_['slowest']}, {v_['slowest_s']} s). "
                             "The result is kept for 15 minutes unless data changes. If it is always slow, run VACUUM (ANALYZE) on the "
                             "Postgres tables: it sets the visibility map so counts can use the index alone.")
            if any(e["prof"] == "whpg" for e in es):
                notes.append(whpg_note())
            if any(e["key"] == "clickhouse" for e in es):
                w_ = chd_cache_info()["warning"]
                if w_:
                    notes.append("ClickHouse: " + w_)
            if any(e["key"] == "clickhouse_qbit" for e in es):
                notes.append("ClickHouse QBit has no index: every query scans every row, reading only the top bits of each number. "
                             "The setting is bits read (1 to 32), so latency grows with the table size, and the lowest settings "
                             "read too little of each number to rank neighbours.")
            for e in es:
                if not plans[e["key"]]["index"]:
                    notes.append(f"{e['label']}: no index was used by the planner "
                                 f"({', '.join(plans[e['key']]['nodes'])}) — that sweep is measuring a "
                                 "sequential scan."
                                 + (" On WarehousePG check that the optimizer setting took effect."
                                    if e["prof"] == "whpg" else ""))
            if cold:
                for key, cs in cleans.items():
                    lab = engines_by_key()[key]["label"]
                    for er in cs["errors"]:
                        notes.append(f"{lab}: cache cleaning — {er}")
                    if cs["shared"] is None and cs["os"] is None and cs["cmd"] is None:
                        notes.append(f"{lab}: nothing could be evicted — this 'cold' run is not cold.")
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
            if job: job["phase"] = "saving"
            run_id = bench_save(
                "sweep", ((body.get("label") or "").strip()[:200] or
                          f"{rows_n:,} rows · k={k} · {'cold' if cold else 'warm'} · {nq} queries") + (body.get("label_suffix") or ""),
                rows_n, dim, k, nq, opclass, "cold" if cold else "warm", passes,
                {"order": [e["key"] for e in es], "exact_ms": exact_avg, "at_floor": at_floor, "notes": notes,
                 "engines": {e["key"]: bench_engine_payload(e, results[e["key"]], plans[e["key"]]["index"]) for e in es},
                 "protocol": {"text": proto, "truth": truth_note, "cache_mode": "cold" if cold else "warm", "passes": passes,
                              "repeat": body.get("_repeat"),
                              "prewarm": prewarmed, "iterative_scan": it, "epsilon": eps,
                              "whpg_session": WH["session"] if any(e["prof"] == "whpg" for e in es) else None}})
            notes.append(truth_note[0].upper() + truth_note[1:] + ".")
            return ({"ok": True, "k": k, "queries": nq, "opclass": opclass, "rows": rows_n,
                            "run_id": run_id, "truth_source": truth_note,
                            "exact_ms": exact_avg, "order": [e["key"] for e in es],
                            "engines": {e["key"]: {"label": e["label"], "results": results[e["key"]],
                                                   "index": plans[e["key"]]["index"], "prof": e["prof"],
                                                   "lists": lists_of(e["key"]), "param": e["param"]}
                                        for e in es},
                            "at_floor": at_floor, "notes": notes,
                            "protocol": {"cache_mode": "cold" if cold else "warm", "passes": passes,
                                         "interleaved": True, "jit": "off", "prewarm": prewarmed,
                                         "clean": cleans, "text": proto,
                                         "whpg_session": WH["session"] if any(e["prof"] == "whpg" for e in es) else None,
                                         "caps": ({k_: caps[k_] for k_ in ("same_host", "os_evict", "shared_evict",
                                                                           "drop_caches_cmd", "pg_major")} if caps else {})},
                            "note": (f"{proto}; recall averaged over {nq} sampled query vectors against one "
                                     "shared exact ground truth; ± is the standard error (recall) and the "
                                     "pass-to-pass deviation (latency).")})
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            if CN:
                CN.close()


    @app.route("/api/cmp/recall", methods=["POST"])
    def cmp_recall():
        """Synchronous form, kept for scripts. The dashboard uses /start and polls — a sweep can run
        for many minutes and no browser connection should have to stay open that long."""
        res = _recall_core(request.get_json(silent=True) or {})
        return jsonify(res), (200 if res.get("ok") else 400)

    def _quiet_stop(body, job):
        """Stop the ClickHouse service so it cannot compete with the engines being measured."""
        if "clickhouse" in (body.get("engines") or []):
            raise ValueError("ClickHouse is one of the engines selected, so it cannot be stopped for this sweep.")
        if not SERVICE_CONTROL:
            raise ValueError("Stopping ClickHouse needs service control. Start the app with VP_ALLOW_SERVICE_CONTROL=1.")
        if not ch_local(CH) or ch_service_state(CH).get("state") != "active":
            return False                                  # nothing to stop: the host is already quiet
        job["phase"], job["current"] = "quiet", "stopping the ClickHouse service for a quiet measurement…"
        cfg = ch_candidate(dict(CH, dbname=CH["database"]))
        CH["quiet"] = True
        _svc_worker("stop", cfg)
        if CH_SVC.get("error"):
            CH["quiet"] = False
            raise RuntimeError("Could not stop ClickHouse: " + str(CH_SVC["error"]))
        return True

    def _quiet_restart(job):
        job["phase"], job["current"] = "quiet", "starting the ClickHouse service again…"
        try:
            _svc_worker("start", ch_candidate(dict(CH, dbname=CH["database"])))
        finally:
            CH["quiet"] = False

    def _sweep_thread(body):
        n = max(1, min(5, int(body.get("repeats") or 1)))
        stopped, ids, res = False, [], {"ok": False, "error": "not started"}
        try:
            if body.get("quiet_host"):
                stopped = _quiet_stop(body, SWEEP)
            import uuid as _uuid
            group = _uuid.uuid4().hex[:8]
            for i in range(n):
                if SWEEP.get("cancel"):
                    break
                SWEEP.update(repeat=i + 1, repeats=n, done=0, total=0, phase="preparing", rep_started=time.time())
                b2 = dict(body, label_suffix=(f" · repeat {i + 1} of {n}" if n > 1 else ""),
                          _repeat={"i": i + 1, "n": n, "group": group} if n > 1 else None)
                res = _recall_core(b2, SWEEP)
                if not res.get("ok"):
                    break
                ids.append(res.get("run_id"))
        except Exception as e:
            res = {"ok": False, "error": str(e)}
        finally:
            if stopped:
                try:
                    _quiet_restart(SWEEP)
                except Exception as e:
                    res = dict(res, ok=False, error=(str(res.get("error") or "") + f" ClickHouse could not be restarted: {e}").strip())
        if SWEEP.get("cancel") and not res.get("ok"):
            saved = f" Repeats already finished were saved (run {', '.join('#' + str(i) for i in ids if i)})." if ids else " Nothing was saved."
            res = {"ok": False, "error": "Sweep cancelled." + saved + " The exact baseline is kept for next time."}
        elif ids and not res.get("ok"):
            res = dict(res, error=str(res.get("error")) + f" Repeats already finished were saved (run {', '.join('#' + str(i) for i in ids if i)}).")
        if res.get("ok"):
            res = dict(res, run_ids=ids)
        SWEEP["result"] = res if res.get("ok") else None
        SWEEP["error"] = None if res.get("ok") else res.get("error")
        SWEEP["run_id"] = res.get("run_id")
        SWEEP["run_ids"] = ids
        SWEEP["phase"] = "done" if res.get("ok") else ("cancelled" if "cancelled" in str(res.get("error")) else "error")
        SWEEP["elapsed"] = round(time.time() - SWEEP["started"], 1)
        SWEEP["running"] = False

    @app.route("/api/cmp/recall/start", methods=["POST"])
    def cmp_recall_start():
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is already running — wait for it."}), 400
        SWEEP.update(running=True, phase="preparing", done=0, total=0, current="", started=time.time(),
                     elapsed=0, error=None, result=None, cancel=False, run_id=None, pids={},
                     repeat=0, repeats=1, run_ids=[])
        threading.Thread(target=_sweep_thread, args=(request.get_json(silent=True) or {},), daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/cmp/recall/status")
    def cmp_recall_status():
        s = dict(SWEEP)
        s.pop("pids", None)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 1)
            s["eta"] = None
            if s["done"] and s["total"]:
                in_rep = max(0.001, time.time() - (s.get("rep_started") or s["started"]))
                per_rep = in_rep / s["done"] * s["total"]                 # this repeat, start to finish
                s["eta"] = round(per_rep - in_rep + (max(1, s.get("repeats") or 1) - max(1, s.get("repeat") or 1)) * per_rep)
        s["pct"] = round(100 * s["done"] / s["total"], 1) if s["total"] else 0
        return jsonify(s)

    @app.route("/api/cmp/recall/cancel", methods=["POST"])
    def cmp_recall_cancel():
        """Flags the job and cancels whatever statement it is inside — an exact scan or a prewarm can
        otherwise run for many minutes before the next check."""
        SWEEP["cancel"] = True
        for prof, pid in list((SWEEP.get("pids") or {}).items()):
            conn = None
            try:
                conn = prof_conn(prof, autocommit=True)
                conn.cursor().execute("SELECT pg_cancel_backend(%s)", (int(pid),))
            except Exception:
                pass
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
        return jsonify({"ok": True})

    # ── identical churn on every table, in both clusters ────────────────────
    def _copy_with_ids(cur, table, ids, rows, prof="pg"):
        out = []
        for i, (topic, cat, title, body, emb) in zip(ids, rows):
            t = title.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            b = body.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            out.append(f"{i}\t{topic}\t{cat}\t{t}\t{b}\t{vec_literal(emb)}\n")
        buf = io.StringIO("".join(out))
        t0 = time.perf_counter()
        cur.copy_expert(f"COPY {qual(table, prof)} (id, topic, category_id, title, body, embedding) "
                        "FROM STDIN", buf)
        return time.perf_counter() - t0

    def _cmp_churn_worker(ins, upd, dele):
        CMP_CHURN.update({"running": True, "phase": "starting", "done": 0,
                          "total": ins + upd + dele, "error": None, "result": None})
        CN = None
        try:
            es = need_engines(None, need_index=False)
            t = {e["key"]: {"insert": 0.0, "update": 0.0, "delete": 0.0, "vacuum": 0.0} for e in es}
            te = truth_engine(es)
            dim = dim_of(te["table"], te["prof"])
            CN = Conns(0)
            rng = np.random.default_rng(int(time.time()) & 0xFFFF)
            pyrng = __import__("random").Random(int(time.time()) & 0xFFFF)
            nxt = 1 + max(int(pscalar(e["prof"], f"SELECT coalesce(max(id),0) FROM {qual(e['table'], e['prof'])}",
                                      default=0) or 0) for e in es)

            if ins:
                CMP_CHURN["phase"] = "insert"
                left = ins
                while left > 0:
                    n = min(1000, left)
                    rows = G["make_batch"](rng, dim, n, pyrng=pyrng)
                    ids = list(range(nxt, nxt + n))
                    for e in es:
                        t[e["key"]]["insert"] += churn_insert(e, CN, ids, rows)
                    nxt += n; left -= n
                    CMP_CHURN["done"] += n
                for e in es:
                    if e["prof"] == "pg":              # WarehousePG tables take explicit ids, no sequence
                        CN.cursor("pg").execute("SELECT setval(pg_get_serial_sequence(%s,'id'), %s, true)",
                                                (f"{schema()}.{e['table']}", nxt - 1))

            def pick(n):
                c = CN.cursor(te["prof"])
                c.execute(f"SELECT id FROM {qual(te['table'], te['prof'])} ORDER BY random() LIMIT %s", (n,))
                return [r[0] for r in c.fetchall()]

            if upd:
                CMP_CHURN["phase"] = "update"
                ids = pick(upd)
                for e in es:
                    t[e["key"]]["update"] += churn_update(e, CN, ids)
                CMP_CHURN["done"] += upd

            if dele:
                CMP_CHURN["phase"] = "delete"
                ids = pick(dele)
                for e in es:
                    t[e["key"]]["delete"] += churn_delete(e, CN, ids)
                CMP_CHURN["done"] += dele
                CMP_CHURN["phase"] = "vacuum"
                for e in es:
                    t[e["key"]]["vacuum"] += churn_vacuum(e, CN)

            rows_after, idx_bytes = {}, {}
            for e in es:
                rows_after[e["key"]] = count_rows(e["table"], e["prof"])
                idx_bytes[e["key"]] = sum(i["bytes"] for i in engine_indexes(e))
            CMP_CHURN["result"] = {
                "order": [e["key"] for e in es], "labels": {e["key"]: e["label"] for e in es},
                "timings": {k_: {p: round(v, 3) for p, v in d.items()} for k_, d in t.items()},
                "rows": rows_after, "index_bytes": idx_bytes,
                "inserts": ins, "updates": upd, "deletes": dele}
            log_query("cmp-churn", f"identical churn: {ins} inserts, {upd} updates, {dele} deletes on "
                      + ", ".join(e["table"] for e in es),
                      ms=sum(sum(d.values()) for d in t.values()) * 1000)
            R = CMP_CHURN["result"]
            bench_save("churn", f"churn +{ins} / ~{upd} / -{dele}", max(R["rows"].values()), dim,
                       None, None, None, None, None, R)
            CMP_CHURN["phase"] = "done"
        except Exception as e:
            CMP_CHURN["error"] = str(e)
            CMP_CHURN["phase"] = "error"
        finally:
            fp_invalidate()
            CMP_CHURN["running"] = False
            if CN:
                CN.close()

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

    # ════════════════════════════════════════════════════════════════════════
    #  CLICKHOUSE — connection, validation, and (opt-in) service control
    # ════════════════════════════════════════════════════════════════════════
    import getpass, shutil, socket, subprocess
    import urllib.error, urllib.parse, urllib.request

    def ch_local(cfg):
        return str(cfg["host"]).lower() in ("localhost", "127.0.0.1", "::1")

    def systemctl_path():
        return os.environ.get("VP_SYSTEMCTL") or shutil.which("systemctl") or "/usr/bin/systemctl"

    def ch_public():
        unit, sc = CH["service"], systemctl_path()
        return {"enabled": CH["enabled"], "host": CH["host"], "port": CH["port"], "user": CH["user"],
                "dbname": CH["database"], "has_password": bool(CH["password"]), "service": unit,
                "local": ch_local(CH), "service_control": SERVICE_CONTROL,
                "app_user": getpass.getuser(), "systemctl": sc,
                "sudoers": f"{getpass.getuser()} ALL=(root) NOPASSWD: {sc} start {unit}, {sc} stop {unit}"}

    def ch_candidate(body):
        host = (body.get("host") or "").strip()
        user = (body.get("user") or "").strip()
        dbn = (body.get("dbname") or "default").strip()
        unit = (body.get("service") or CH["service"]).strip()
        if not host or not user:
            raise ValueError("host and user are required")
        try:
            port = int(body.get("port") or 8123)
        except ValueError:
            raise ValueError("port must be a number")
        if not (1 <= port <= 65535):
            raise ValueError("port must be 1–65535")
        if not _IDENT_RE.match(dbn):
            raise ValueError("database must be letters, digits and underscores")
        if not _UNIT_RE.match(unit):
            raise ValueError("service name may only contain letters, digits and _ . @ -")
        pw = body.get("password")
        if body.get("clear_password"):
            pw = ""
        elif pw in (None, ""):
            pw = CH.get("password", "")
        return {"host": host, "port": port, "user": user, "password": pw, "database": dbn, "service": unit}

    def ch_query(cfg, sql, timeout=10, database=None, params=None):
        """One statement over ClickHouse's HTTP interface. Errors come back as a readable message."""
        qs = {}
        if database:
            qs["database"] = database
        for k_, v_ in (params or {}).items():
            qs["param_" + k_] = v_
        url = f"http://{cfg['host']}:{int(cfg['port'])}/" + (("?" + urllib.parse.urlencode(qs)) if qs else "")
        req = urllib.request.Request(url, data=sql.encode(), method="POST",
                                     headers={"X-ClickHouse-User": cfg["user"],
                                              "X-ClickHouse-Key": cfg.get("password", "")})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            body_ = e.read().decode("utf-8", "replace").strip()
            code = e.headers.get("X-ClickHouse-Exception-Code")
            first = body_.splitlines()[0] if body_ else f"HTTP {e.code}"
            raise RuntimeError(first[:300] + (f"  [error {code}]" if code and f"{code}" not in first else ""))

    def _ch_with(cfg, sql, settings, timeout=20):
        """Same as ch_query, with settings passed as URL parameters (they apply to this request only)."""
        qs = {k_: v_ for k_, v_ in settings.items()}
        url = f"http://{cfg['host']}:{int(cfg['port'])}/?" + urllib.parse.urlencode(qs)
        req = urllib.request.Request(url, data=sql.encode(), method="POST",
                                     headers={"X-ClickHouse-User": cfg["user"], "X-ClickHouse-Key": cfg.get("password", "")})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            body_ = e.read().decode("utf-8", "replace").strip()
            code = e.headers.get("X-ClickHouse-Exception-Code")
            first = body_.splitlines()[0] if body_ else f"HTTP {e.code}"
            raise RuntimeError(first[:300] + (f"  [error {code}]" if code and f"{code}" not in first else ""))

    def qbit_probe(cfg):
        """Make a tiny QBit table, query it at 32 bits (exact), and drop it. Returns (ok, message, settings)."""
        try:
            names = ch_query(cfg, "SELECT name FROM system.settings WHERE name IN ('allow_experimental_qbit_type','enable_qbit_type')").split()
        except Exception as e:
            return None, f"not checked: {e}", {}
        en = {n_: 1 for n_ in names}
        tbl = f"{cfg['database']}._vp_qbit_{int(time.time()) % 100000}"
        try:
            _ch_with(cfg, f"CREATE TABLE {tbl} (id UInt32, v QBit(Float32, 8)) ENGINE = MergeTree ORDER BY id", en, 20)
            _ch_with(cfg, f"INSERT INTO {tbl} SELECT number, [number, number+1, number+2, number+3, number+4, number+5, number+6, number+7] FROM numbers(200)", en, 20)
            top = _ch_with(cfg, f"SELECT id FROM {tbl} ORDER BY L2DistanceTransposed(v, [10.,11.,12.,13.,14.,15.,16.,17.], 32) LIMIT 3", en, 20).split()
            ok = bool(top) and top[0] == "10"
            return ok, ("QBit(Float32, 8) created; L2DistanceTransposed at 32 bits found id 10 nearest" if ok else f"query ran but returned {top}"), en
        except Exception as e:
            return False, str(e)[:240], en
        finally:
            try:
                _ch_with(cfg, f"DROP TABLE IF EXISTS {tbl}", {}, 20)
            except Exception:
                pass

    def ch_ping(cfg, timeout=3):
        t0 = time.perf_counter()
        with urllib.request.urlopen(f"http://{cfg['host']}:{int(cfg['port'])}/ping", timeout=timeout) as r:
            txt = r.read().decode().strip()
        return txt, round((time.perf_counter() - t0) * 1000, 1)

    def ch_service_state(cfg):
        if not ch_local(cfg):
            return {"state": None, "note": "remote host — its service is not managed from here"}
        try:
            r = subprocess.run([systemctl_path(), "is-active", cfg["service"]],
                               capture_output=True, text=True, timeout=8)
            return {"state": (r.stdout.strip() or r.stderr.strip() or "unknown")}
        except FileNotFoundError:
            return {"state": None, "note": "systemctl not found on this machine"}
        except Exception as e:
            return {"state": None, "note": str(e)}

    def ch_port_open(cfg, timeout=2.5):
        t0 = time.perf_counter()
        try:
            socket.create_connection((cfg["host"], int(cfg["port"])), timeout).close()
            return True, round((time.perf_counter() - t0) * 1000, 1), None
        except Exception as e:
            return False, round((time.perf_counter() - t0) * 1000, 1), str(e)

    def ch_validate(cfg):
        """Five layers, each answering a different 'why not': is the service up, does the port answer,
        is it really ClickHouse, does the login work, can it do what this dashboard needs."""
        steps, info = [], {}
        def add(name, ok, detail, hint=None, required=True):
            steps.append({"name": name, "ok": ok, "detail": detail, "hint": hint, "required": required})
        unit = cfg["service"]
        ss = ch_service_state(cfg)
        if ss["state"] is None:
            add("systemd service", None, ss.get("note", "not checked"), required=False)
        elif ss["state"] == "active":
            add("systemd service", True, f"{unit} is active", required=False)
        else:
            add("systemd service", False, f"{unit} is {ss['state']}",
                f"Press ▶ Start service, or run:  sudo systemctl start {unit}", required=False)
        ok, ms, err = ch_port_open(cfg)
        add("TCP port", ok, f"{cfg['host']}:{cfg['port']} accepts connections ({ms} ms)" if ok
            else f"{cfg['host']}:{cfg['port']} — {err}",
            None if ok else f"Nothing is listening. Check the service is running and that http_port in "
                            f"/etc/clickhouse-server/config.xml is {cfg['port']} (8123 by default).")
        if not ok:
            return {"ok": False, "steps": steps, "info": info}
        try:
            txt, pms = ch_ping(cfg)
            add("HTTP /ping", txt == "Ok.", f"answered “{txt}” in {pms} ms" if txt == "Ok."
                else f"answered “{txt[:60]}” — this does not look like ClickHouse",
                None if txt == "Ok." else "The port belongs to something else; ClickHouse's HTTP port is 8123, its native port 9000.")
            info["ping_ms"] = pms
        except Exception as e:
            add("HTTP /ping", False, str(e), "The port is open but did not answer like ClickHouse over HTTP.")
            return {"ok": False, "steps": steps, "info": info}
        try:
            row = ch_query(cfg, "SELECT version(), currentUser(), uptime(), timezone()").strip().split("\t")
            info.update(version=row[0], user=row[1], uptime_s=int(row[2]), timezone=row[3])
            add("Login & query", True, f"ClickHouse {row[0]} as “{row[1]}”, up {int(row[2])} s")
        except Exception as e:
            msg = str(e)
            hint = ("The user or password is wrong — type the password into the Password box and press Validate again. "
                    "If ClickHouse was installed with a password, it is stored (as a hash) in "
                    "/etc/clickhouse-server/users.d/default-password.xml; other users are in users.xml or users.d/." if "516" in msg or "194" in msg or "uthentication" in msg
                    else "The server answered but refused the query.")
            add("Login & query", False, msg, hint)
            return {"ok": False, "steps": steps, "info": info}
        try:
            n = ch_query(cfg, "SELECT count() FROM system.databases WHERE name = {db:String}",
                         params={"db": cfg["database"]}).strip()
            add("Database", n == "1", f"“{cfg['database']}” exists" if n == "1" else f"“{cfg['database']}” does not exist",
                None if n == "1" else f"Create it:  clickhouse-client -q \"CREATE DATABASE {cfg['database']}\"")
        except Exception as e:
            add("Database", False, str(e))
        try:
            rows = [r.split("\t") for r in ch_query(
                cfg, "SELECT name, value FROM system.settings WHERE name IN "
                     "('enable_vector_similarity_index','allow_experimental_vector_similarity_index')").strip().splitlines() if r]
            on = [n_ for n_, v_ in rows if v_ in ("1", "true")]
            info["vector_settings"] = {n_: v_ for n_, v_ in rows}
            if on:
                add("Vector similarity index", True, f"available ({on[0]} = 1)", required=False)
            elif rows:
                add("Vector similarity index", False, ", ".join(f"{n_} = {v_}" for n_, v_ in rows),
                    f"Enable it per session with  SET {rows[0][0]} = 1  (the dashboard will do this itself).", required=False)
            else:
                add("Vector similarity index", False, "this ClickHouse version has no vector_similarity setting",
                    "Vector indexes need a recent ClickHouse (24.8 or later, beta from 25.x). Upgrade the server.", required=False)
        except Exception as e:
            add("Vector similarity index", None, f"not checked: {e}", required=False)
        # The settings only say a feature is switched on. This proves it works on THIS build: make a tiny
        # table with a vector index, search it, confirm the planner used the index, and drop it again.
        db_ok = any(s_["name"] == "Database" and s_["ok"] for s_ in steps)
        vs = info.get("vector_settings") or {}
        if db_ok and vs:
            tbl = f"{cfg['database']}._vp_smoke_{int(time.time()) % 100000}"
            opts = {k_: 1 for k_ in vs}                      # only settings this build knows about
            def run(sql, timeout=20):
                return ch_query(cfg, sql, timeout=timeout, params=None) if not opts else _ch_with(cfg, sql, opts, timeout)
            try:
                run(f"CREATE TABLE {tbl} (id UInt32, v Array(Float32), "
                    f"INDEX vi v TYPE vector_similarity('hnsw', 'L2Distance', 4) GRANULARITY 100000000) "
                    f"ENGINE = MergeTree ORDER BY id")
                run(f"INSERT INTO {tbl} SELECT number, [number, number + 1, number + 2, number + 3] FROM numbers(300)")
                top = run(f"SELECT id FROM {tbl} ORDER BY L2Distance(v, [10., 11., 12., 13.]) LIMIT 3").split()
                plan = run(f"EXPLAIN indexes = 1 SELECT id FROM {tbl} ORDER BY L2Distance(v, [10., 11., 12., 13.]) LIMIT 3")
                used = "vector_similarity" in plan or " vi" in plan or "Name: vi" in plan
                add("Vector index smoke test", bool(top) and top[0] == "10" and used,
                    f"built an HNSW index on 300 rows, nearest to [10,11,12,13] was id {top[0] if top else '?'}"
                    + (", and the planner used the index" if used else ", but the planner did NOT use the index"),
                    None if used else "The index was created but not used. Show me the EXPLAIN output of a vector query on this server.",
                    required=False)
            except Exception as e:
                msg = str(e)
                hint = ("This user may not create tables here. Grant it, or use a different database/user." if "ACCESS_DENIED" in msg or "ccess denied" in msg or "Not enough privileges" in msg or "497" in msg
                        else "This build did not accept the vector_similarity index syntax used here. Send me this message and your ClickHouse version.")
                add("Vector index smoke test", False, msg[:240], hint, required=False)
            finally:
                try:
                    run(f"DROP TABLE IF EXISTS {tbl}")
                except Exception:
                    pass
        if db_ok:
            ok_q, msg_q, qs_ = qbit_probe(cfg)
            add("QBit type", ok_q, msg_q, None if ok_q else "QBit needs ClickHouse 25.10 or later. Without it the ClickHouse QBit engine is unavailable; HNSW is unaffected.",
                required=False)
        req_ok = all(s["ok"] for s in steps if s["required"] and s["ok"] is not None)
        return {"ok": req_ok, "steps": steps, "info": info}

    @app.route("/api/ch")
    def ch_get():
        return jsonify({"ok": True, **ch_public()})

    @app.route("/api/ch/test", methods=["POST"])
    def ch_test():
        try:
            cfg = ch_candidate(request.get_json(silent=True) or {})
            return jsonify({"ok": True, **ch_validate(cfg), "config": {k_: cfg[k_] for k_ in ("host", "port", "database", "service")}})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/ch/apply", methods=["POST"])
    def ch_apply():
        body = request.get_json(silent=True) or {}
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        try:
            cfg = ch_candidate(body)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        res = ch_validate(cfg)
        if not res["ok"]:                         # never switch to something that does not work
            bad = next((s for s in res["steps"] if s["required"] and s["ok"] is False), None)
            return jsonify({"ok": False, "error": (f"{bad['name']}: {bad['detail']}" if bad else "validation failed"),
                            **res}), 400
        CH.update(cfg)
        CH["enabled"] = True
        saved = False
        if body.get("save"):
            payload = {k_: cfg[k_] for k_ in ("host", "port", "user", "database", "service")}
            payload["enabled"] = True
            if body.get("save_password"):
                payload["password"] = cfg["password"]
            try:
                write_saved({"clickhouse": payload})
                saved = True
            except Exception as e:
                res["save_error"] = str(e)
        return jsonify({"ok": True, "saved": saved, **res, **ch_public()})

    @app.route("/api/ch/disconnect", methods=["POST"])
    def ch_disconnect():
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        CH["enabled"] = False
        try:
            with open(CONN_FILE) as fh:
                saved = json.load(fh)
            if isinstance(saved.get("clickhouse"), dict):
                saved["clickhouse"]["enabled"] = False
                write_saved({"clickhouse": saved["clickhouse"]})
        except Exception:
            pass
        return jsonify({"ok": True, **ch_public()})

    def load_saved_ch():
        try:
            with open(CONN_FILE) as fh:
                s = json.load(fh).get("clickhouse")
        except Exception:
            return
        if not s:
            return
        for k_ in ("host", "user", "database", "service"):
            if s.get(k_):
                CH[k_] = s[k_]
        if s.get("port"):
            CH["port"] = int(s["port"])
        if s.get("password"):
            CH["password"] = s["password"]
        CH["enabled"] = bool(s.get("enabled"))

    # ── service control: exactly two commands, only when the operator allowed it ──
    def _svc_fail(msg, hint=None):
        CH_SVC.update(error=msg, hint=hint, phase="error")

    def _svc_step(name, ok, detail):
        CH_SVC["steps"].append({"name": name, "ok": ok, "detail": detail})

    def _svc_worker(action, cfg):
        unit = cfg["service"]
        CH_SVC.update(running=True, action=action, phase="running", steps=[], started=time.time(),
                      elapsed=0, error=None, hint=None, result=None)
        try:
            cmd = [systemctl_path(), action, unit]
            if os.geteuid() != 0:
                cmd = ["sudo", "-n"] + cmd                       # never prompts: fails instead
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            took = round(time.perf_counter() - t0, 1)
            if r.returncode != 0:
                err = (r.stderr or r.stdout).strip() or f"exit status {r.returncode}"
                _svc_step(" ".join(cmd), False, err)
                low = err.lower()
                if "password is required" in low or "no tty" in low or "not in the sudoers" in low or "may not run sudo" in low:
                    _svc_fail("sudo needs a password for this command, and the app cannot type one.",
                              "Allow exactly these two commands without a password. As root:\n"
                              f"  echo '{ch_public()['sudoers']}' > /etc/sudoers.d/vectorplus-clickhouse\n"
                              "  chmod 440 /etc/sudoers.d/vectorplus-clickhouse && visudo -cf /etc/sudoers.d/vectorplus-clickhouse\n"
                              "Then check as the app user:  sudo -n -l")
                elif "unit" in low and "not found" in low:
                    _svc_fail(f"There is no systemd unit called “{unit}”.",
                              "List the real name with:  systemctl list-unit-files | grep -i clickhouse")
                elif "command not found" in low or "no such file" in low:
                    _svc_fail("systemctl or sudo was not found.", f"Looked for {systemctl_path()}; set VP_SYSTEMCTL to its full path.")
                else:
                    _svc_fail(err[:300])
                return
            _svc_step(" ".join(cmd), True, f"returned in {took} s")
            deadline = time.time() + (90 if action == "start" else 40)
            if action == "start":
                while time.time() < deadline:
                    try:
                        txt, ms = ch_ping(cfg, timeout=2)
                        if txt == "Ok.":
                            _svc_step("waiting for /ping", True, f"ClickHouse answered {round(time.time() - CH_SVC['started'], 1)} s after the start command")
                            break
                    except Exception:
                        pass
                    time.sleep(0.6)
                else:
                    _svc_step("waiting for /ping", False, "the service started but never answered on the HTTP port")
                    _svc_fail(f"{unit} started but {cfg['host']}:{cfg['port']} did not answer within 90 s.",
                              f"Look at the log:  journalctl -u {unit} -n 50 --no-pager   or   /var/log/clickhouse-server/clickhouse-server.err.log")
                    return
            else:
                while time.time() < deadline:
                    st = ch_service_state(cfg)["state"]
                    if st != "active" and not ch_port_open(cfg, 1)[0]:
                        _svc_step("confirming it stopped", True, f"service {st}, port closed")
                        break
                    time.sleep(0.6)
                else:
                    _svc_step("confirming it stopped", False, "still answering after 40 s")
                    _svc_fail(f"{unit} is still running after the stop command.", f"Check:  systemctl status {unit}")
                    return
            CH_SVC["result"] = ch_validate(cfg) if action == "start" else {"ok": True}
            CH_SVC["phase"] = "done"
        except subprocess.TimeoutExpired:
            _svc_fail("systemctl did not return within 120 s.", f"Check:  systemctl status {unit}")
        except Exception as e:
            _svc_fail(str(e))
        finally:
            CH_SVC["elapsed"] = round(time.time() - CH_SVC["started"], 1)
            CH_SVC["running"] = False

    @app.route("/api/ch/service", methods=["POST"])
    def ch_service():
        body = request.get_json(silent=True) or {}
        action = body.get("action")
        if action == "status":
            cfg = ch_candidate(dict(CH, dbname=CH["database"], **{k_: v_ for k_, v_ in body.items() if k_ in ("host", "port", "user", "service")}))
            return jsonify({"ok": True, **ch_service_state(cfg)})
        if action not in ("start", "stop"):
            return jsonify({"ok": False, "error": "action must be start, stop or status"}), 400
        if not SERVICE_CONTROL:
            return jsonify({"ok": False, "error": "Service control is switched off.",
                            "hint": "Start the app with  VP_ALLOW_SERVICE_CONTROL=1  to allow it. It is off by default because "
                                    "this app has no login: anyone who can reach its port could otherwise start and stop the service."}), 403
        try:
            cfg = ch_candidate(dict(body, dbname=body.get("dbname") or CH["database"]))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not ch_local(cfg):
            return jsonify({"ok": False, "error": "Only a service on this machine can be started or stopped from here."}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        threading.Thread(target=_svc_worker, args=(action, cfg), daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/ch/service/status")
    def ch_service_status():
        s = dict(CH_SVC)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 1)
        return jsonify(s)

    # ════════════════════════════════════════════════════════════════════════
    #  CLICKHOUSE AS A COMPARISON ENGINE
    # ════════════════════════════════════════════════════════════════════════
    # ClickHouse speaks HTTP and its own SQL, not the PostgreSQL protocol. ChCursor is a small adapter
    # with the same shape as a psycopg2 cursor, so the search, sweep and churn code above runs on it
    # unchanged. Everything that is ClickHouse-specific lives here.

    CH_ALLOWED = {"hnsw_candidate_list_size_for_search", "vector_search_filter_strategy",
                  "vector_search_with_rescoring", "max_threads"}
    _CH_ENABLE = {"settings": None, "qbit": None}
    _VEC_LIT = re.compile(r"\[[-+0-9eE.,\s]*\]")

    def ch_enable_settings():
        """The 'turn the vector index on' settings this particular build knows — asked once."""
        if _CH_ENABLE["settings"] is None:
            try:
                names = ch_query(CH, "SELECT name FROM system.settings WHERE name IN "
                                     "('enable_vector_similarity_index','allow_experimental_vector_similarity_index')").split()
                _CH_ENABLE["settings"] = {n_: 1 for n_ in names}
            except Exception:
                return {}
        return dict(_CH_ENABLE["settings"])

    def qbit_settings():
        """The settings this build uses to switch the QBit type on (the name changed between versions)."""
        if _CH_ENABLE["qbit"] is None:
            try:
                names = ch_query(CH, "SELECT name FROM system.settings WHERE name IN "
                                     "('allow_experimental_qbit_type','enable_qbit_type')").split()
                _CH_ENABLE["qbit"] = {n_: 1 for n_ in names}
            except Exception:
                return {}
        return dict(_CH_ENABLE["qbit"])

    def ch_rows(sql, timeout=30):
        raw = _ch_with(CH, sql, {"default_format": "JSONEachRow", "output_format_json_quote_64bit_integers": 0}, timeout)
        return [json.loads(l_) for l_ in raw.splitlines() if l_.strip()]

    class ChCursor:
        """A psycopg2-shaped cursor over ClickHouse's HTTP interface. Settings sent with SET are kept and
        applied to later queries; ones ClickHouse does not know (enable_indexscan, jit, …) are ignored."""
        def __init__(self):
            self.settings, self._data, self.description, self.rowcount = {}, [], None, -1
            self.qbit_bits = 16
        def mogrify(self, sql, params):
            if params is None:
                return sql
            if not isinstance(params, (tuple, list)):
                params = (params,)
            parts = sql.split("%s")
            if len(parts) - 1 != len(params):
                raise ValueError("parameter count does not match the placeholders")
            parts = [p_.replace("__QBIT_BITS__", str(int(self.qbit_bits))) for p_ in parts]
            out = [parts[0]]
            for p_, rest in zip(params, parts[1:]):
                t_ = str(p_)
                if not _VEC_LIT.fullmatch(t_):
                    raise ValueError("only vector literals may be passed as query parameters")
                out += [t_, rest]
            return "".join(out)
        def execute(self, sql, params=None):
            st = sql.strip()
            m = re.match(r"^SET\s+([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.+)$", st, re.I | re.S)
            if m:
                if m.group(1) == "qbit_bits":                  # not a ClickHouse setting: the precision argument of the QBit functions
                    self.qbit_bits = max(1, min(64, int(m.group(2))))
                elif m.group(1) in CH_ALLOWED:
                    self.settings[m.group(1)] = m.group(2).strip().strip("'")
                return
            if re.match(r"^SELECT\s+pg_backend_pid", st, re.I):
                raise RuntimeError("ClickHouse has no backend pid")
            full = self.mogrify(st, params) if params is not None else st.replace("__QBIT_BITS__", str(int(self.qbit_bits)))
            sets = dict(ch_enable_settings(), **qbit_settings(), **self.settings)
            sets.update({"default_format": "JSONCompact", "output_format_json_quote_64bit_integers": 0})
            raw = _ch_with(CH, full, sets, 300)
            doc = json.loads(raw) if raw.strip() else {"meta": [], "data": []}
            self._data = [tuple(r_) for r_ in doc.get("data", [])]
            self.description = [(m_["name"],) for m_ in doc.get("meta", [])]
            self.rowcount = len(self._data)
        def fetchall(self):
            return list(self._data)
        def fetchone(self):
            return self._data[0] if self._data else None
        def close(self):
            pass

    class ChConn:
        autocommit, notices = True, []
        def cursor(self, *a, **k):
            return ChCursor()
        def close(self):
            pass
        def commit(self):
            pass
        def rollback(self):
            pass

    def ch_error():
        """None when ClickHouse is off or answering, else why not."""
        if not CH["enabled"]:
            return None
        try:
            ch_ping(CH, 4)
            return None
        except Exception as e:
            return str(e).replace("<urlopen error ", "").rstrip(">")

    def ann_sql_ch(table, opclass, limit, category_id):
        fn = {"vector_l2_ops": "L2Distance", "vector_cosine_ops": "cosineDistance"}.get(opclass)
        if table == CHQ_TABLE and fn:
            fn_q = fn + "Transposed"                       # L2DistanceTransposed / cosineDistanceTransposed
            where_q = f"WHERE category_id = {int(category_id)}\n" if category_id not in (None, "", "any") else ""
            return (f"SELECT id, topic, category_id, title, {fn_q}(embedding, %s, __QBIT_BITS__) AS distance\n"
                    f"FROM {qual(table, 'ch')}\n{where_q}ORDER BY {fn_q}(embedding, %s, __QBIT_BITS__)\nLIMIT {int(limit)}")
        if not fn:
            raise ValueError("ClickHouse's vector index supports L2 and cosine distance here "
                             f"(operator class vector_l2_ops or vector_cosine_ops), not {opclass}")
        where = f"WHERE category_id = {int(category_id)}\n" if category_id not in (None, "", "any") else ""
        return (f"SELECT id, topic, category_id, title, {fn}(embedding, %s) AS distance\n"
                f"FROM {qual(table, 'ch')}\n{where}ORDER BY {fn}(embedding, %s)\nLIMIT {int(limit)}")

    def chd_plan(cur, sql, params):
        """Did ClickHouse's planner use the vector index? Same question plan_of asks of PostgreSQL."""
        if f"`{CHQ_TABLE}`" in sql:                          # QBit: no index, every row is read (only the top bits of each)
            return {"nodes": [f"ReadFromMergeTree (QBit scan, {cur.qbit_bits} bits read, no index)"], "index": "qbit"}
        cur.execute("EXPLAIN indexes = 1 " + cur.mogrify(sql, params))
        text = "\n".join(r_[0] for r_ in cur.fetchall())
        used = "vector_similarity" in text or "Name: vi" in text
        return {"nodes": ["ReadFromMergeTree" + (" + vector_similarity index" if used else " (full scan)")],
                "index": "vi" if used else None}

    def chd_indexes():
        db, tb = CH["database"], CH_TABLE
        try:
            rows = ch_rows(f"SELECT name, type_full FROM system.data_skipping_indices "
                           f"WHERE database = '{db}' AND table = '{tb}' AND type = 'vector_similarity'")
        except Exception:
            try:
                rows = ch_rows(f"SELECT name, '' AS type_full FROM system.data_skipping_indices "
                               f"WHERE database = '{db}' AND table = '{tb}' AND type = 'vector_similarity'")
            except Exception:
                return []
        if not rows:
            return []
        try:
            b = int(ch_rows(f"SELECT sum(secondary_indices_compressed_bytes) AS b FROM system.parts "
                            f"WHERE database = '{db}' AND table = '{tb}' AND active")[0]["b"] or 0)
        except Exception:
            b = 0
        try:
            mem_b = int(ch_rows(f"SELECT sum(secondary_indices_uncompressed_bytes) AS u FROM system.parts "
                                f"WHERE database = '{db}' AND table = '{tb}' AND active")[0]["u"] or 0)
        except Exception:
            mem_b = 0
        out = []
        for r_ in rows:
            tf = r_.get("type_full") or ""
            m = re.search(r"vector_similarity\('hnsw',\s*'(\w+)',\s*(\d+)(?:,\s*'(\w+)',\s*(\d+),\s*(\d+))?", tf)
            out.append({"name": r_["name"], "bytes": b, "size": human_bytes(b), "valid": True, "lists": None, "memory_bytes": mem_b,
                        "memory": human_bytes(mem_b),
                        "opclass": ("vector_cosine_ops" if m and m.group(1) == "cosineDistance" else "vector_l2_ops") if m else None,
                        "quantization": (m.group(3) if m and m.group(3) else "bf16"),
                        "m": int(m.group(4)) if m and m.group(4) else None,
                        "ef_construction": int(m.group(5)) if m and m.group(5) else None, "definition": tf})
        return out

    def _tsv_rows(ids, rows):
        out = []
        for i_, (topic, cat, title, body, emb) in zip(ids, rows):
            t_ = title.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            b_ = body.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            out.append(f"{i_}\t{topic}\t{cat}\t{t_}\t{b_}\t{vec_literal(emb)}\n")
        return "".join(out).encode("utf-8")

    def ch_insert_tsv(data, timeout=900):
        q_ = f"INSERT INTO {qual(CH_TABLE, 'ch')} (id, topic, category_id, title, body, embedding) FORMAT TabSeparated"
        url = (f"http://{CH['host']}:{int(CH['port'])}/?" +
               urllib.parse.urlencode({"query": q_, "max_insert_block_size": 1048576}))
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"X-ClickHouse-User": CH["user"], "X-ClickHouse-Key": CH.get("password", "")})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                r.read()
        except urllib.error.HTTPError as e:
            body_ = e.read().decode("utf-8", "replace").strip()
            raise RuntimeError((body_.splitlines()[0] if body_ else f"HTTP {e.code}")[:300])

    def chd_insert_rows(ids, rows):
        t0 = time.perf_counter()
        ch_insert_tsv(_tsv_rows(ids, rows))
        return time.perf_counter() - t0

    CH_LONG = {"send_progress_in_http_headers": 1, "http_headers_progress_interval_ms": 10000}

    def chd_ddl(dim):
        return (f"CREATE TABLE {qual(CH_TABLE, 'ch')} (id UInt64, topic String, category_id Int32, title String, "
                f"body String, embedding Array(Float32)) ENGINE = MergeTree ORDER BY id")

    # ── status ──────────────────────────────────────────────────────────────
    def chd_advice(rows):
        return {"quantization": "bf16", "m": 16, "ef_construction": 128,
                "note": "ClickHouse builds one HNSW graph per data part, so the table is merged into a single part "
                        "after the sync. Quantization trades recall for memory: f32 is exact, bf16 is its default, "
                        "i8 and b1 are smaller and coarser. ef_search is hnsw_candidate_list_size_for_search "
                        "(default 256); the dashboard sets it from the shared probes value."}

    def chd_cache_info():
        """ClickHouse keeps vector indexes in an in-memory cache. An index larger than that cache is reloaded
        on every query, which would make 'search latency' mostly measure loading."""
        db, tb = CH["database"], CH_TABLE
        out = {"cache_bytes": None, "index_memory_bytes": 0, "mem_total": None, "warning": None}
        try:
            c_ = ch_rows("SELECT value FROM system.server_settings WHERE name = 'vector_similarity_index_cache_size'")
            out["cache_bytes"] = int(float(c_[0]["value"])) if c_ else None
        except Exception:
            pass
        try:
            out["index_memory_bytes"] = int(ch_rows(f"SELECT sum(secondary_indices_uncompressed_bytes) AS u FROM system.parts "
                                                    f"WHERE database = '{db}' AND table = '{tb}' AND active")[0]["u"] or 0)
        except Exception:
            pass
        try:
            out["mem_total"] = int(float(ch_rows("SELECT value FROM system.asynchronous_metrics WHERE metric = 'OSMemoryTotal'")[0]["value"]))
        except Exception:
            pass
        u, cache, mem = out["index_memory_bytes"], out["cache_bytes"], out["mem_total"]
        if u and cache is not None and u > cache:
            out["warning"] = (f"The vector index is about {human_bytes(u)} in memory but ClickHouse's vector index cache is only "
                              f"{human_bytes(cache)} (vector_similarity_index_cache_size), so queries keep reloading it from disk and "
                              "the latency measured here is mostly loading time. Raise that setting in config.xml (it is picked up "
                              "without a restart) to at least the index size, and make sure the machine has the memory.")
        elif u and mem and u > 0.8 * mem:
            out["warning"] = (f"The vector index is about {human_bytes(u)} in memory, most of this machine's {human_bytes(mem)} of RAM; "
                              "expect swapping or out-of-memory errors.")
        return out

    def chq_indexes():
        """QBit has no index. The column itself is what is searched, so it is reported as one pseudo-index, sized by the column."""
        try:
            r_ = ch_rows(f"SELECT type, data_compressed_bytes AS c FROM system.columns WHERE database = '{CH['database']}' "
                         f"AND table = '{CHQ_TABLE}' AND name = 'embedding'")
        except Exception:
            return []
        if not r_:
            return []
        b = int(r_[0]["c"] or 0)
        return [{"name": "qbit", "bytes": b, "size": human_bytes(b), "valid": True, "lists": None, "opclass": None,
                 "quantization": r_[0]["type"], "m": None, "ef_construction": None, "memory_bytes": 0, "memory": "no index",
                 "definition": r_[0]["type"]}]

    @app.route("/api/chd/status")
    def chd_status():
        out = {"enabled": CH["enabled"], "table": CH_TABLE, "database": CH["database"], "vp_table": vp_table(),
               "sync": CHD_SYNC, "build": {k_: v_ for k_, v_ in CHD_BUILD.items()}}
        if not CH["enabled"]:
            return jsonify(out)
        err = ch_error()
        if err:
            out["error"] = err
            return jsonify(out)
        out["connected"] = True
        out["exists"] = exists(CH_TABLE, "ch")
        out["vp_exists"] = exists(vp_table())
        src = fingerprint_cached(vp_table()) if out["vp_exists"] else None
        out["source_rows"] = src[0] if src else rows_est(vp_table())
        if out["exists"]:
            fp = fingerprint_cached(CH_TABLE, "ch")
            out["rows"], out["rows_exact"] = (fp[0] if fp else rows_est(CH_TABLE, "ch")), bool(fp)
            out["in_sync"] = (fp == src) if (fp and src) else None
            out["dim"] = dim_of(CH_TABLE, "ch")
            try:
                s_ = ch_rows(f"SELECT count() AS parts, sum(data_compressed_bytes) AS data FROM system.parts "
                             f"WHERE database = '{CH['database']}' AND table = '{CH_TABLE}' AND active")[0]
                out["parts"], out["data_bytes"], out["data_size"] = int(s_["parts"]), int(s_["data"] or 0), human_bytes(int(s_["data"] or 0))
            except Exception:
                pass
            out["indexes"] = chd_indexes()
            if out["indexes"]:
                out["cache"] = chd_cache_info()
        qx = exists(CHQ_TABLE, "ch")
        out["qbit"] = {"exists": qx, "table": CHQ_TABLE, "build": dict(CHQ_BUILD)}
        if qx:
            qi = chq_indexes()
            fq = fingerprint_cached(CHQ_TABLE, "ch")
            out["qbit"].update(rows=(fq[0] if fq else rows_est(CHQ_TABLE, "ch")), rows_exact=bool(fq), dim=dim_of(CHQ_TABLE, "ch"),
                               size=(qi[0]["size"] if qi else None), bytes=(qi[0]["bytes"] if qi else 0), type=(qi[0]["quantization"] if qi else None),
                               in_sync=((fq == fingerprint_cached(CH_TABLE, "ch")) if (fq and fingerprint_cached(CH_TABLE, "ch")) else None))
        out["advisor"] = chd_advice(out.get("rows") or 0)
        out["version"] = _ch_ver()
        return jsonify(out)

    def _ch_ver():
        try:
            return ch_query(CH, "SELECT version()", timeout=3).strip()
        except Exception:
            return None

    # ── sync: the same rows, the same ids, streamed straight from Postgres ──
    def _chd_sync_worker():
        CHD_SYNC.update(running=True, phase="preparing", done=0, total=0, started=time.time(), elapsed=0,
                        error=None, result=None)
        fp_invalidate()
        pgc = None
        try:
            vp = vp_table()
            dim = dim_of(vp)
            if not dim:
                raise RuntimeError(f"{vp} does not exist or has no embedding column")
            pgc = get_conn(autocommit=True)
            pcur = pgc.cursor()
            pcur.execute("SET statement_timeout = 0")
            pcur.execute(f"SELECT count(*), coalesce(min(id),0), coalesce(max(id),0) FROM {qual(vp)}")
            total, lo, hi = pcur.fetchone()
            CHD_SYNC["total"] = int(total)
            ch_query(CH, f"DROP TABLE IF EXISTS {qual(CH_TABLE, 'ch')}")
            ch_query(CH, chd_ddl(dim))
            step = max(1000, min(20000, int(30e6 / (dim * 9 + 200))))      # about 30 MB per request
            t0 = time.perf_counter()
            CHD_SYNC["phase"] = "copying"
            at = lo
            while at <= hi:
                buf = io.BytesIO()
                pcur.copy_expert(f"COPY (SELECT id, topic, category_id, title, body, embedding::text FROM {qual(vp)} "
                                 f"WHERE id >= {int(at)} AND id < {int(at) + step}) TO STDOUT", buf)
                data = buf.getvalue()
                if data:
                    ch_insert_tsv(data)
                    CHD_SYNC["done"] += data.count(b"\n")
                CHD_SYNC["elapsed"] = round(time.time() - CHD_SYNC["started"], 1)
                at += step
            copy_s = round(time.perf_counter() - t0, 3)
            CHD_SYNC["phase"] = "merging parts"
            t1 = time.perf_counter()
            _ch_with(CH, f"OPTIMIZE TABLE {qual(CH_TABLE, 'ch')} FINAL", CH_LONG, 7200)
            merge_s = round(time.perf_counter() - t1, 3)
            fp_invalidate()
            fpc, fpp = fingerprint(CH_TABLE, "ch", fresh=True), fingerprint(vp, "pg", fresh=True)
            res = {"rows": CHD_SYNC["done"], "copy_seconds": copy_s, "merge_seconds": merge_s,
                   "rows_per_sec": int(CHD_SYNC["done"] / copy_s) if copy_s else 0, "dim": dim,
                   "identical": fpc == fpp, "clickhouse": list(fpc) if fpc else None, "postgres": list(fpp) if fpp else None}
            CHD_SYNC["result"] = res
            store_insert("loads", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target=CH_TARGET,
                         rows=res["rows"], dim=dim, seed=None, seconds=copy_s, rows_per_sec=res["rows_per_sec"],
                         generator=f"sync from {vp}")
            if not res["identical"]:
                raise RuntimeError(f"ClickHouse holds {fpc[0]:,} rows but Postgres has {fpp[0]:,}, or the ids differ — "
                                   "the copy is not identical; re-run the sync.")
            CHD_SYNC["phase"] = "done"
        except Exception as e:
            CHD_SYNC["error"] = str(e)
            CHD_SYNC["phase"] = "error"
        finally:
            fp_invalidate()
            CHD_SYNC["elapsed"] = round(time.time() - (CHD_SYNC["started"] or time.time()), 1)
            CHD_SYNC["running"] = False
            if pgc:
                try: pgc.close()
                except Exception: pass

    @app.route("/api/chd/sync", methods=["POST"])
    def chd_sync():
        if not CH["enabled"]:
            return jsonify({"ok": False, "error": "ClickHouse is not connected — press Connect on the Setup page first."}), 400
        err = ch_error()
        if err:
            return jsonify({"ok": False, "error": f"ClickHouse cannot be reached: {err}"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        if not exists(vp_table()) or not count_rows(vp_table()):
            return jsonify({"ok": False, "error": f"{vp_table()} is missing or empty on Postgres — load it first."}), 400
        threading.Thread(target=_chd_sync_worker, daemon=True).start()
        return jsonify({"ok": True})

    @app.route("/api/chd/sync/status")
    def chd_sync_status():
        s = dict(CHD_SYNC)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 1)
        s["pct"] = round(100 * s["done"] / s["total"], 1) if s["total"] else 0
        return jsonify(s)

    # ── index build: ALTER … MATERIALIZE INDEX is a background mutation, so it is polled, not awaited ──
    def _chd_build_worker(sql_add, sql_mat, opts, dim, rows, replace):
        CHD_BUILD.update(running=True, phase="preparing", started=time.time(), elapsed=0, error=None,
                         result=None, sql=sql_add + ";\n" + sql_mat, parts_to_do=None)
        try:
            en = ch_enable_settings()
            tb = qual(CH_TABLE, "ch")
            if replace:
                _ch_with(CH, f"ALTER TABLE {tb} DROP INDEX IF EXISTS vi", dict(en, mutations_sync=2), 3600)
            _ch_with(CH, sql_add, en, 60)
            CHD_BUILD["phase"] = "building"
            t0 = time.perf_counter()
            _ch_with(CH, sql_mat, en, 60)                       # returns at once; the work is a mutation
            while True:
                m_ = ch_rows(f"SELECT count() AS n, sum(parts_to_do) AS p, anyIf(latest_fail_reason, latest_fail_reason != '') AS why "
                             f"FROM system.mutations WHERE database = '{CH['database']}' AND table = '{CH_TABLE}' AND is_done = 0")[0]
                CHD_BUILD["parts_to_do"] = int(m_["p"] or 0)
                CHD_BUILD["elapsed"] = round(time.time() - CHD_BUILD["started"], 1)
                if m_.get("why"):
                    raise RuntimeError("ClickHouse could not build the index: " + str(m_["why"])[:300])
                if int(m_["n"]) == 0:
                    break
                time.sleep(0.7)
            elapsed = time.perf_counter() - t0
            idx = chd_indexes()
            size_b = idx[0]["bytes"] if idx else 0
            rec = {"name": "vi", "seconds": round(elapsed, 3), "rows": rows, "rows_per_sec": int(rows / elapsed) if elapsed > 0 else 0,
                   "bytes": size_b, "size": human_bytes(size_b), "bytes_per_row": round(size_b / rows, 1) if rows else 0,
                   "opts": opts, "sql": sql_add}
            rec["id"] = store_insert(
                "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target=CH_TARGET, label="ClickHouse",
                engine="clickhouse", am="vector_similarity", dsn_db=CH["database"], rows=rows, dim=dim, lists=None,
                m=opts["m"], ef_construction=opts["ef_construction"], opclass=opts["opclass"], concurrently=0,
                index_name="vi", build_seconds=round(elapsed, 3), rows_per_sec=round(rows / elapsed, 1) if elapsed > 0 else 0,
                index_bytes=size_b, bytes_per_row=round(size_b / rows, 2) if rows else 0, maint_work_mem=None,
                parallel_workers=None, pg_version=None, ext_version=_ch_ver(), notices=None,
                note=f"quantization={opts['quantization']}")
            CHD_BUILD["result"] = rec
            CHD_BUILD["phase"] = "done"
        except Exception as e:
            CHD_BUILD["error"] = str(e)
            CHD_BUILD["phase"] = "error"
        finally:
            CHD_BUILD["elapsed"] = round(time.time() - (CHD_BUILD["started"] or time.time()), 1)
            CHD_BUILD["running"] = False

    @app.route("/api/chd/index/build/start", methods=["POST"])
    def chd_build_start():
        if not CH["enabled"]:
            return jsonify({"ok": False, "error": "ClickHouse is not connected."}), 400
        err = ch_error()
        if err:
            return jsonify({"ok": False, "error": f"ClickHouse cannot be reached: {err}"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        body = request.get_json(silent=True) or {}
        opclass = body.get("opclass", "vector_l2_ops")
        fn = {"vector_l2_ops": "L2Distance", "vector_cosine_ops": "cosineDistance"}.get(opclass)
        if not fn:
            return jsonify({"ok": False, "error": "ClickHouse's vector index supports L2 and cosine here (vector_l2_ops, vector_cosine_ops)."}), 400
        quant = body.get("quantization") or "bf16"
        if quant not in ("f32", "f16", "bf16", "i8", "b1"):
            return jsonify({"ok": False, "error": "quantization must be f32, f16, bf16, i8 or b1"}), 400
        try:
            m_ = int(body.get("m") or 16)
            efc = int(body.get("ef_construction") or 128)
            if not (2 <= m_ <= 200) or not (4 <= efc <= 4000):
                raise ValueError("m must be 2–200 and ef_construction 4–4000")
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not exists(CH_TABLE, "ch"):
            return jsonify({"ok": False, "error": f"{CH_TABLE} does not exist in ClickHouse — sync it first."}), 400
        rows, dim = count_rows(CH_TABLE, "ch"), dim_of(CH_TABLE, "ch")
        if not rows:
            return jsonify({"ok": False, "error": f"{CH_TABLE} is empty — sync it first."}), 400
        tb = qual(CH_TABLE, "ch")
        sql_add = (f"ALTER TABLE {tb} ADD INDEX vi embedding TYPE vector_similarity('hnsw', '{fn}', {dim}, "
                   f"'{quant}', {m_}, {efc}) GRANULARITY 100000000")
        sql_mat = f"ALTER TABLE {tb} MATERIALIZE INDEX vi"
        warnings = []
        parts = ch_rows(f"SELECT count() AS n FROM system.parts WHERE database = '{CH['database']}' AND table = '{CH_TABLE}' AND active")[0]["n"]
        if int(parts) > 1:
            warnings.append(f"The table has {parts} data parts, and ClickHouse builds one HNSW graph per part — more parts "
                            "means slower queries. Re-run the sync (it merges to one part), or run OPTIMIZE TABLE … FINAL.")
        warnings.append("Build time is the time for the index to be materialized over the existing data; "
                        "ClickHouse loads each graph into memory on the first query that needs it.")
        opts = {"m": m_, "ef_construction": efc, "quantization": quant, "opclass": opclass}
        threading.Thread(target=_chd_build_worker, args=(sql_add, sql_mat, opts, dim, rows, body.get("replace", True)), daemon=True).start()
        return jsonify({"ok": True, "sql": sql_add + ";\n" + sql_mat, "warnings": warnings})

    def _chq_build_worker(dim, rows):
        CHQ_BUILD.update(running=True, phase="creating the table", started=time.time(), elapsed=0, error=None, result=None)
        try:
            en = qbit_settings()
            tb, src = qual(CHQ_TABLE, "ch"), qual(CH_TABLE, "ch")
            ddl = (f"CREATE TABLE {tb} (id UInt64, topic String, category_id Int32, title String, body String, "
                   f"embedding QBit(Float32, {dim})) ENGINE = MergeTree ORDER BY id")
            cvt = f"INSERT INTO {tb} SELECT id, topic, category_id, title, body, embedding FROM {src}"
            CHQ_BUILD["sql"] = ddl + ";\n" + cvt
            _ch_with(CH, f"DROP TABLE IF EXISTS {tb}", {}, 600)
            _ch_with(CH, ddl, en, 60)
            CHQ_BUILD["phase"] = "converting the vectors to bit planes"
            t0 = time.perf_counter()
            _ch_with(CH, cvt, dict(en, **CH_LONG), 7200)
            conv = time.perf_counter() - t0
            CHQ_BUILD["phase"] = "merging parts"
            t1 = time.perf_counter()
            _ch_with(CH, f"OPTIMIZE TABLE {tb} FINAL", CH_LONG, 7200)
            merge = time.perf_counter() - t1
            fp_invalidate()
            fq, fc = fingerprint(CHQ_TABLE, "ch", fresh=True), fingerprint(CH_TABLE, "ch", fresh=True)
            idx = chq_indexes()
            size_b = idx[0]["bytes"] if idx else 0
            rec = {"name": "qbit", "seconds": round(conv, 3), "merge_seconds": round(merge, 3), "rows": rows,
                   "rows_per_sec": int(rows / conv) if conv > 0 else 0, "bytes": size_b, "size": human_bytes(size_b),
                   "bytes_per_row": round(size_b / rows, 1) if rows else 0, "identical": fq == fc, "sql": CHQ_BUILD["sql"]}
            rec["id"] = store_insert(
                "builds", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), target=CH_TARGET, label="ClickHouse QBit",
                engine="clickhouse_qbit", am="clickhouse_qbit", dsn_db=CH["database"], rows=rows, dim=dim, lists=None, m=None,
                ef_construction=None, opclass="vector_l2_ops", concurrently=0, index_name="qbit", build_seconds=round(conv, 3),
                rows_per_sec=round(rows / conv, 1) if conv > 0 else 0, index_bytes=size_b,
                bytes_per_row=round(size_b / rows, 2) if rows else 0, maint_work_mem=None, parallel_workers=None, pg_version=None,
                ext_version=_ch_ver(), notices=None, note=f"QBit(Float32, {dim}); merge to one part took {merge:.1f}s (not counted)")
            CHQ_BUILD["result"] = rec
            if not rec["identical"]:
                raise RuntimeError(f"The QBit table holds {fq[0]:,} rows but the Array table holds {fc[0]:,}, or the ids differ — rebuild it.")
            CHQ_BUILD["phase"] = "done"
        except Exception as e:
            CHQ_BUILD["error"] = str(e)
            CHQ_BUILD["phase"] = "error"
        finally:
            fp_invalidate()
            CHQ_BUILD["elapsed"] = round(time.time() - (CHQ_BUILD["started"] or time.time()), 1)
            CHQ_BUILD["running"] = False

    @app.route("/api/chd/qbit/build/start", methods=["POST"])
    def chq_build_start():
        if not CH["enabled"]:
            return jsonify({"ok": False, "error": "ClickHouse is not connected."}), 400
        err = ch_error()
        if err:
            return jsonify({"ok": False, "error": f"ClickHouse cannot be reached: {err}"}), 400
        what = busy()
        if what:
            return jsonify({"ok": False, "error": f"A {what} is running — wait for it."}), 400
        if not exists(CH_TABLE, "ch") or not count_rows(CH_TABLE, "ch"):
            return jsonify({"ok": False, "error": f"{CH_TABLE} is missing or empty in ClickHouse. The QBit table is made from it, so sync it first."}), 400
        ok_q, msg_q, _ = qbit_probe(CH)
        if not ok_q:
            return jsonify({"ok": False, "error": "This ClickHouse cannot use QBit: " + str(msg_q)}), 400
        dim, rows = dim_of(CH_TABLE, "ch"), count_rows(CH_TABLE, "ch")
        threading.Thread(target=_chq_build_worker, args=(dim, rows), daemon=True).start()
        return jsonify({"ok": True, "warnings": [
            "QBit stores each number as bit planes, so a Float32 vector is kept in full and the precision is chosen per query. "
            "There is no index: every query scans every row, so latency grows with the table size."]})

    @app.route("/api/chd/qbit/build/status")
    def chq_build_status():
        s_ = dict(CHQ_BUILD)
        if s_["running"] and s_["started"]:
            s_["elapsed"] = round(time.time() - s_["started"], 1)
        return jsonify(s_)

    @app.route("/api/chd/qbit/drop", methods=["POST"])
    def chq_drop():
        if not CH["enabled"]:
            return jsonify({"ok": False, "error": "ClickHouse is not connected."}), 400
        try:
            _ch_with(CH, f"DROP TABLE IF EXISTS {qual(CHQ_TABLE, 'ch')}", {}, 600)
            fp_invalidate()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/chd/index/build/status")
    def chd_build_status():
        s = dict(CHD_BUILD)
        if s["running"] and s["started"]:
            s["elapsed"] = round(time.time() - s["started"], 1)
        return jsonify(s)

    @app.route("/api/chd/index/list")
    def chd_index_list():
        hist = store_query("SELECT * FROM builds WHERE target = ? ORDER BY id DESC LIMIT 50", (CH_TARGET,))
        for h_ in hist:
            h_["at"] = (h_.get("ts") or "")[5:16]
            h_["size"] = human_bytes(h_.get("index_bytes"))
        return jsonify({"ok": True, "rows": chd_indexes() if CH["enabled"] and ch_error() is None else [], "history": hist})

    @app.route("/api/chd/index/drop", methods=["POST"])
    def chd_index_drop():
        if not CH["enabled"]:
            return jsonify({"ok": False, "error": "ClickHouse is not connected."}), 400
        try:
            _ch_with(CH, f"ALTER TABLE {qual(CH_TABLE, 'ch')} DROP INDEX IF EXISTS vi", {"mutations_sync": 2}, 600)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ── churn: the same rows in and out, ClickHouse's way (mutations) ────────
    def churn_insert(e, CN, ids, rows):
        if e["prof"] == "ch" and e["table"] == CHQ_TABLE:
            t0 = time.perf_counter()
            _ch_with(CH, f"INSERT INTO {qual(CHQ_TABLE, 'ch')} SELECT id, topic, category_id, title, body, embedding "
                         f"FROM {qual(CH_TABLE, 'ch')} WHERE id IN ({','.join(str(int(i_)) for i_ in ids)})",
                     dict(qbit_settings(), max_query_size=200000000), 1800)
            return time.perf_counter() - t0
        if e["prof"] == "ch":
            return chd_insert_rows(ids, rows)
        return _copy_with_ids(CN.cursor(e["prof"]), e["table"], ids, rows, e["prof"])

    def churn_update(e, CN, ids):
        t0 = time.perf_counter()
        if e["prof"] == "ch":
            _ch_with(CH, f"ALTER TABLE {qual(e['table'], 'ch')} UPDATE body = concat(body, ' revised') "
                         f"WHERE id IN ({','.join(str(int(i_)) for i_ in ids)})",
                     {"mutations_sync": 2, "max_query_size": 200000000}, 3600)
        else:
            CN.cursor(e["prof"]).execute(f"UPDATE {qual(e['table'], e['prof'])} SET body = body || ' revised' "
                                         "WHERE id = ANY(%s)", (ids,))
        return time.perf_counter() - t0

    def churn_delete(e, CN, ids):
        t0 = time.perf_counter()
        if e["prof"] == "ch":
            _ch_with(CH, f"ALTER TABLE {qual(e['table'], 'ch')} DELETE WHERE id IN ({','.join(str(int(i_)) for i_ in ids)})",
                     {"mutations_sync": 2, "max_query_size": 200000000}, 3600)
        else:
            CN.cursor(e["prof"]).execute(f"DELETE FROM {qual(e['table'], e['prof'])} WHERE id = ANY(%s)", (ids,))
        return time.perf_counter() - t0

    def churn_vacuum(e, CN):
        t0 = time.perf_counter()
        if e["prof"] == "ch":
            _ch_with(CH, f"OPTIMIZE TABLE {qual(e['table'], 'ch')} FINAL", CH_LONG, 7200)
        else:
            CN.cursor(e["prof"]).execute(f"VACUUM {qual(e['table'], e['prof'])}")
        return time.perf_counter() - t0

    # ════════════════════════════════════════════════════════════════════════
    #  BENCHMARK RUNS — every sweep and churn is kept, so 2M and 10M can be ranked later
    # ════════════════════════════════════════════════════════════════════════
    # A run is self-contained: the engines, their settings and results, the build
    # that produced each index, the protocol and the environment. The benchmark page
    # ranks from these, so a later re-load or re-index of the tables cannot change
    # what a saved run says. They live in the same SQLite file as the build history.

    BENCH_SCHEMA = """CREATE TABLE IF NOT EXISTS bench_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT UNIQUE, ts TEXT, kind TEXT,
        label TEXT, notes TEXT, rows INTEGER, dim INTEGER, k INTEGER, queries INTEGER,
        opclass TEXT, cache_mode TEXT, passes INTEGER,
        starred INTEGER DEFAULT 0, hidden INTEGER DEFAULT 0, env TEXT, payload TEXT);
    CREATE TABLE IF NOT EXISTS bench_truth (
        key TEXT PRIMARY KEY, ts TEXT, k INTEGER, nq INTEGER, rows INTEGER, exact_ms TEXT, truth TEXT)"""
    BENCH_COLS = ("uid", "ts", "kind", "label", "notes", "rows", "dim", "k", "queries", "opclass",
                  "cache_mode", "passes", "starred", "hidden", "env", "payload")

    _bench_lock = threading.Lock()

    def bench_db(fn):
        with (G.get("_STORE_LOCK") or _bench_lock):
            c = G["store_conn"]()
            try:
                c.executescript(BENCH_SCHEMA)
                out = fn(c)
                c.commit()
                return out
            finally:
                c.close()

    def bench_env():
        """What the numbers were measured on — never credentials."""
        env = {"saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "postgres": {"version": scalar("SELECT current_setting('server_version')"),
                            "shared_buffers": scalar("SELECT current_setting('shared_buffers')"),
                            "block_size": scalar("SELECT current_setting('block_size')"),
                            "maintenance_work_mem": scalar("SELECT current_setting('maintenance_work_mem')"),
                            "max_parallel_maintenance_workers": scalar(
                                "SELECT current_setting('max_parallel_maintenance_workers')"),
                            "extensions": {r["extname"]: r["extversion"] for r in q(
                                "SELECT extname, extversion FROM pg_extension WHERE extname IN "
                                "('edb_vectorplus','vector','vchord','pg_buffercache','pg_prewarm')")}}}
        if WH["enabled"] and not wh_error():
            env["whpg"] = {"version": (pscalar("whpg", "SELECT version()") or "")[:120],
                           "segments": WH.get("segments"), "session": list(WH["session"]),
                           "extensions": {r["extname"]: r["extversion"] for r in pq(
                               "whpg", "SELECT extname, extversion FROM pg_extension "
                                       "WHERE extname IN ('edb_vectorplus','vector')")}}
        if CH["enabled"] and not ch_error():
            try:
                ix_ = chd_indexes()
                env["clickhouse"] = {"version": ch_query(CH, "SELECT version()", timeout=3).strip(), "database": CH["database"],
                                     "index": ({k_: ix_[0].get(k_) for k_ in ("quantization", "m", "ef_construction", "opclass")} if ix_ else None)}
            except Exception:
                pass
        return env

    def bench_save(kind, label, rows, dim, k, queries, opclass, cache_mode, passes, payload):
        """Never let bookkeeping break a measurement that already succeeded."""
        try:
            import uuid
            uid = uuid.uuid4().hex
            vals = (uid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kind, label, "", rows, dim, k,
                    queries, opclass, cache_mode, passes, 0, 0, json.dumps(bench_env()), json.dumps(payload))
            return bench_db(lambda c: c.execute(
                f"INSERT INTO bench_runs ({','.join(BENCH_COLS)}) VALUES ({','.join('?' * len(BENCH_COLS))})",
                vals).lastrowid)
        except Exception as ex:
            print(f"  [bench] could not save run: {ex}")
            return None

    def bench_engine_payload(e, results, plan_index):
        """Everything about one engine that the ranking needs, frozen at sweep time."""
        idx = engine_indexes(e)
        ix = next((i for i in idx if i["name"] == plan_index), idx[0] if idx else None)
        b = store_query("SELECT * FROM builds WHERE target = ? AND engine = ? AND index_name = ? "
                        "ORDER BY id DESC LIMIT 1", (e["target"], e["am"], plan_index)) if plan_index else []
        if not b:
            b = store_query("SELECT * FROM builds WHERE target = ? AND engine = ? ORDER BY id DESC LIMIT 1",
                            (e["target"], e["am"]))
        b = b[0] if b else {}
        return {"label": e["label"], "prof": e["prof"], "table": e["table"], "am": e["am"],
                "base": e["base"], "param": e["param"], "index": plan_index,
                "lists": (ix or {}).get("lists"), "m": (ix or {}).get("m"),
                "ef_construction": (ix or {}).get("ef_construction"),
                "index_bytes": int((ix or {}).get("bytes") or 0) or None,
                "build": {k_: b.get(k_) for k_ in ("build_seconds", "rows_per_sec", "index_bytes",
                                                   "bytes_per_row", "maint_work_mem", "parallel_workers",
                                                   "lists", "m", "ef_construction", "ts", "rows") } if b else None,
                "results": results}

    def bench_row(r, with_payload=False):
        d = {k_: r.get(k_) for k_ in ("id", "uid", "ts", "kind", "label", "notes", "rows", "dim", "k",
                                      "queries", "opclass", "cache_mode", "passes", "starred", "hidden")}
        d["env"] = json.loads(r["env"]) if r.get("env") else {}
        if with_payload:
            d["payload"] = json.loads(r["payload"]) if r.get("payload") else {}
        else:
            p = json.loads(r["payload"]) if r.get("payload") else {}
            d["engines"] = p.get("order") or list((p.get("labels") or {}).keys())
        return d

    @app.route("/api/bench/runs")
    def bench_runs():
        kind = request.args.get("kind")
        def fn(c):
            sql = "SELECT * FROM bench_runs" + (" WHERE kind = ?" if kind else "") + " ORDER BY id DESC LIMIT 500"
            return [dict(r) for r in c.execute(sql, (kind,) if kind else ()).fetchall()]
        rows = bench_db(fn)
        return jsonify({"ok": True, "runs": [bench_row(r) for r in rows], "results_db": G.get("RESULTS_DB")})

    @app.route("/api/bench/run/<int:rid>")
    def bench_run_one(rid):
        rows = bench_db(lambda c: [dict(r) for r in c.execute("SELECT * FROM bench_runs WHERE id = ?", (rid,)).fetchall()])
        if not rows:
            return jsonify({"ok": False, "error": f"No run #{rid}"}), 404
        return jsonify({"ok": True, "run": bench_row(rows[0], with_payload=True)})

    @app.route("/api/bench/runs/payloads")
    def bench_payloads():
        ids = [int(x) for x in (request.args.get("ids") or "").split(",") if x.strip().isdigit()][:40]
        if not ids:
            return jsonify({"ok": True, "runs": []})
        rows = bench_db(lambda c: [dict(r) for r in c.execute(
            f"SELECT * FROM bench_runs WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()])
        return jsonify({"ok": True, "runs": [bench_row(r, with_payload=True) for r in rows]})

    @app.route("/api/bench/run/<int:rid>/update", methods=["POST"])
    def bench_run_update(rid):
        body = request.get_json(silent=True) or {}
        sets, vals = [], []
        for f in ("label", "notes"):
            if f in body:
                sets.append(f"{f} = ?"); vals.append(str(body[f] or "")[:2000])
        for f in ("starred", "hidden"):
            if f in body:
                sets.append(f"{f} = ?"); vals.append(1 if body[f] else 0)
        if not sets:
            return jsonify({"ok": False, "error": "nothing to update"}), 400
        bench_db(lambda c: c.execute(f"UPDATE bench_runs SET {', '.join(sets)} WHERE id = ?", vals + [rid]))
        return jsonify({"ok": True})

    @app.route("/api/bench/run/<int:rid>/delete", methods=["POST"])
    def bench_run_delete(rid):
        bench_db(lambda c: c.execute("DELETE FROM bench_runs WHERE id = ?", (rid,)))
        return jsonify({"ok": True})

    @app.route("/api/bench/export")
    def bench_export():
        ids = [int(x) for x in (request.args.get("ids") or "").split(",") if x.strip().isdigit()]
        def fn(c):
            if ids:
                return [dict(r) for r in c.execute(
                    f"SELECT * FROM bench_runs WHERE id IN ({','.join('?' * len(ids))}) ORDER BY id", ids).fetchall()]
            return [dict(r) for r in c.execute("SELECT * FROM bench_runs ORDER BY id").fetchall()]
        runs = [bench_row(r, with_payload=True) for r in bench_db(fn)]
        resp = jsonify({"format": "vp-bench-1", "exported": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "runs": runs})
        resp.headers["Content-Disposition"] = (
            f'attachment; filename="vectorplus-benchmark-{datetime.now().strftime("%Y%m%d-%H%M%S")}.json"')
        return resp

    @app.route("/api/bench/import", methods=["POST"])
    def bench_import():
        """Load runs exported from another machine or cluster. Runs already present
        (same uid) are skipped, so importing the same file twice is harmless."""
        body = request.get_json(silent=True) or {}
        runs = body.get("runs")
        if not isinstance(runs, list) or body.get("format", "vp-bench-1") != "vp-bench-1":
            return jsonify({"ok": False, "error": "not a vectorplus benchmark export (expected format vp-bench-1)"}), 400
        import uuid
        added = skipped = 0
        def fn(c):
            nonlocal added, skipped
            for r in runs:
                if not isinstance(r, dict) or not isinstance(r.get("payload"), dict):
                    skipped += 1; continue
                uid = str(r.get("uid") or uuid.uuid4().hex)
                if c.execute("SELECT 1 FROM bench_runs WHERE uid = ?", (uid,)).fetchone():
                    skipped += 1; continue
                vals = (uid, r.get("ts"), r.get("kind") or "sweep", str(r.get("label") or "imported")[:200],
                        str(r.get("notes") or "")[:2000], r.get("rows"), r.get("dim"), r.get("k"),
                        r.get("queries"), r.get("opclass"), r.get("cache_mode"), r.get("passes"),
                        1 if r.get("starred") else 0, 0, json.dumps(r.get("env") or {}), json.dumps(r["payload"]))
                c.execute(f"INSERT INTO bench_runs ({','.join(BENCH_COLS)}) VALUES ({','.join('?' * len(BENCH_COLS))})", vals)
                added += 1
        bench_db(fn)
        return jsonify({"ok": True, "added": added, "skipped": skipped})

    @app.route("/api/cmp/categories")
    def cmp_categories():
        try:
            rows = q(f"SELECT category_id, count(*) AS n FROM {qual(vp_table())} "
                     f"GROUP BY 1 ORDER BY 1")
        except Exception:
            rows = []
        return jsonify({"ok": True, "rows": rows})

    @app.errorhandler(Exception)
    def _json_errors(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException) or not request.path.startswith("/api/"):
            return e
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

    # saved connection is applied last, so it overrides the environment defaults
    # but is itself overridden by explicit command-line flags (they become the
    # parser defaults after this runs, then argparse applies whatever was passed).
    load_saved()
    print(f"  [vc_engine] registered — VectorChord table '{VC_TABLE}', "
          f"connection source: {CONN_META['source']}")
