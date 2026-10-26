"""Every number in the deck comes from here, computed from the five vp-bench-1 exports in ./data."""
import glob, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {}
for f in glob.glob(os.path.join(HERE, "data", "vectorplus-benchmark-*.json")):
    r = json.load(open(f))["runs"][0]
    RUNS[r["id"]] = r

QUIET_2M = (3, 4, 5, 6)       # same queries, same exact baseline, same indexes; run 6 extends the ladder to 256 probes
GIB = 2 ** 30


def _eng(rid, key):
    return RUNS[rid]["payload"]["engines"].get(key)


def consolidate(run_ids, keys):
    """Per engine and setting, the record with the lowest latency across the repeat runs.
    Interference only ever adds latency, and recall is identical between these runs."""
    out = {}
    for key in keys:
        cands = [_eng(r, key) for r in run_ids if _eng(r, key)]
        base = cands[-1]
        res = []
        for x in base["results"]:
            best = None
            for e in cands:
                y = next((z for z in e["results"] if z["param"] == x["param"]), None)
                if y and (best is None or y["ms"] < best["ms"]):
                    best = y
            res.append(dict(best))
        out[key] = dict(label=base["label"], prof=base["prof"], build=base["build"],
                        index_bytes=base["index_bytes"], results=res, lists=base.get("lists"))
    return out


def single(rid):
    out = {}
    for key, e in RUNS[rid]["payload"]["engines"].items():
        out[key] = dict(label=e["label"], prof=e["prof"], build=e["build"], index_bytes=e["index_bytes"],
                        results=e["results"], lists=e.get("lists"))
    return out


D1 = single(1)
D2 = consolidate(QUIET_2M, ["ivfplus", "vchordrq", "ivfflat", "w_ivfplus", "w_ivfflat", "clickhouse"])

WEIGHTS = dict(lat=40, p95=10, rec=10, build=15, size=15)      # maintenance (10) was not measured


def score(D, bar):
    rows = {}
    for k, e in D.items():
        ok = [x for x in e["results"] if x["recall"] >= bar]
        pick = min(ok, key=lambda x: x["ms"]) if ok else None
        rows[k] = dict(key=k, label=e["label"], lat=pick["ms"] if pick else None, p95=pick["p95_ms"] if pick else None,
                       param=pick["param"] if pick else None, rec=max(x["recall"] for x in e["results"]),
                       build=e["build"]["build_seconds"], size=e["index_bytes"], qual=bool(pick), pts={})
    tot = sum(WEIGHTS.values())
    for m, low in (("lat", 1), ("p95", 1), ("build", 1), ("size", 1), ("rec", 0)):
        vals = [r[m] for r in rows.values() if r[m] is not None]
        best = min(vals) if low else max(vals)
        for r in rows.values():
            v = r[m]
            r["pts"][m] = 0 if v is None else 100 * WEIGHTS[m] / tot * (best / v if low else v / best)
    for r in rows.values():
        r["score"] = sum(r["pts"].values())
    return sorted(rows.values(), key=lambda r: (not r["qual"], -r["score"]))


def lat_at(e, target):
    """Latency at a recall level, interpolated in log-latency along the measured curve."""
    pts = sorted({(x["recall"], x["ms"]) for x in e["results"]})
    env = sorted({(r, min(m for rr, m in pts if rr >= r)) for r, _ in pts})
    if target < env[0][0] or target > env[-1][0]:
        return None
    for (r0, m0), (r1, m1) in zip(env, env[1:]):
        if r0 <= target <= r1:
            if r1 == r0:
                return m1
            t = (target - r0) / (r1 - r0)
            return math.exp(math.log(m0) + t * (math.log(m1) - math.log(m0)))
    return env[-1][1]


def at(D, key, param):
    return next(x for x in D[key]["results"] if str(x["param"]) == str(param))


def spikes(rid, ref=5, factor=1.5):
    keys = ["ivfplus", "vchordrq", "ivfflat", "w_ivfplus", "w_ivfflat"]
    n = big = 0
    for k in keys:
        for x in _eng(rid, k)["results"]:
            y = next(z for z in _eng(ref, k)["results"] if z["param"] == x["param"])
            n += 1
            big += x["ms"] / y["ms"] > factor
    return big, n


if __name__ == "__main__":
    for r in score(D2, 0.90):
        print(f"{r['label']:26} {'DNQ' if not r['qual'] else '   '} {r['score']:5.1f}")
    print(spikes(3), spikes(4))


def agreement(run_a, run_b, keys=("ivfplus", "vchordrq", "ivfflat", "w_ivfflat")):
    """Latency ratio run_b / run_a at every setting both runs measured, for the engines both contain."""
    out = []
    for k in keys:
        a = {str(x["param"]): x["ms"] for x in _eng(run_a, k)["results"]}
        b = {str(x["param"]): x["ms"] for x in _eng(run_b, k)["results"]}
        out += [b[p] / a[p] for p in a if p in b]
    return out
