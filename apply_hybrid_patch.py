#!/usr/bin/env python3
"""
apply_hybrid_patch.py — splice hybrid_search_patch.py into vectorplus_api.py.

Replaces the whole HYBRID SEARCH section: from its banner comment down to (but
not including) the FILTERING banner. Writes a timestamped backup first and
refuses to write anything that does not compile.

    python3 apply_hybrid_patch.py                    # apply
    python3 apply_hybrid_patch.py --dry-run          # show what would change
    python3 apply_hybrid_patch.py --revert           # restore newest backup
"""

import argparse
import ast
import datetime
import glob
import os
import shutil
import sys

TARGET = "vectorplus_api.py"
PATCH  = "hybrid_search_patch.py"

START_MARK = "#  HYBRID SEARCH  (vector + full text, fused with RRF)"
END_MARK   = "#  FILTERING"
BANNER     = "# " + "\u2550" * 78


def die(msg):
    print(f"  ✗ {msg}")
    sys.exit(1)


def find_section(lines):
    """Line indices [start, end) of the hybrid section, banner included."""
    try:
        s = next(i for i, l in enumerate(lines) if l.rstrip() == START_MARK)
    except StopIteration:
        die(f"could not find the marker {START_MARK!r} in {TARGET}. "
            "Has this file already been patched by hand?")
    # step back over the banner line above the title
    while s > 0 and lines[s - 1].startswith("# \u2550"):
        s -= 1

    try:
        e = next(i for i, l in enumerate(lines[s:], s)
                 if l.rstrip() == END_MARK)
    except StopIteration:
        die(f"found the hybrid section but not the {END_MARK!r} banner after "
            "it — cannot tell where the section ends.")
    while e > 0 and lines[e - 1].startswith("# \u2550"):
        e -= 1
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--patch", default=PATCH)
    a = ap.parse_args()

    if a.revert:
        backups = sorted(glob.glob(a.target + ".bak-*"))
        if not backups:
            die("no backup found")
        shutil.copy2(backups[-1], a.target)
        print(f"  ✓ restored {backups[-1]} -> {a.target}")
        return

    for f in (a.target, a.patch):
        if not os.path.exists(f):
            die(f"{f} not found — run this from the demo directory")

    lines = open(a.target).read().splitlines(keepends=True)
    s, e = find_section(lines)
    print(f"  hybrid section: lines {s+1}–{e} ({e-s} lines)")

    old = "".join(lines[s:e])
    if "candidates\", 200" in old or 'body.get("candidates", 200)' in old:
        print("  ! this section already looks patched — nothing to do")
        return
    if 'LIMIT %s""", (qv, qv, k))' not in old and "(qv, qv, k)" not in old:
        print("  ! the section does not look like the original "
              "(no `(qv, qv, k)` display query). Check before proceeding.")

    patch = open(a.patch).read().rstrip("\n") + "\n\n\n"
    new_src = "".join(lines[:s]) + patch + "".join(lines[e:])

    try:
        ast.parse(new_src)
    except SyntaxError as ex:
        die(f"the patched file would not compile: {ex}")

    if a.dry_run:
        print(f"  dry run: would replace {e-s} lines with {len(patch.splitlines())}")
        print(f"  removed head: {lines[s].rstrip()[:70]}")
        print(f"  removed tail: {lines[e-1].rstrip()[:70]}")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{a.target}.bak-{stamp}"
    shutil.copy2(a.target, backup)
    open(a.target, "w").write(new_src)

    print(f"  ✓ backup    {backup}")
    print(f"  ✓ patched   {a.target}")

    check = open(a.target).read()
    for want, why in [
        ('body.get("candidates", 200)', "candidate depth defaults to 200"),
        ("(qv, qv, n)",                 "vector leg runs at candidate depth"),
        ('"returned": len(fts_all)',    "leg depth reported to the UI"),
        ("overlap_topk",                "overlap measured at candidate depth"),
        ("ORDER BY score DESC, d.id",   "deterministic final ordering"),
    ]:
        print(("  ✓ " if want in check else "  ✗ ") + why)

    print("\n  Now restart the API — Flask runs with debug=False, so editing")
    print("  the file on its own changes nothing:")
    print("     pkill -f vectorplus_api.py")
    print("     python3 vectorplus_api.py --corpus bfsi --app-port 5066 ...")


if __name__ == "__main__":
    main()
