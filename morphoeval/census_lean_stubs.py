#!/usr/bin/env python3
"""Census: which morphoeval/lean/loops/*.lean are overnight-template stubs?

A file is a STUB iff it contains the exact generator line
`def law1_invariant (a b : Int) : Int := a * b` — the overnight generators
(morphoeval/run_overnight_*_1000.py) emit the same three arithmetic lemmas
(product positivity, `p + q - q = p`, a constant threshold by `decide`) for
every slot, whatever law the header comment names. Stub files compile with
0 `sorry`s but formalise nothing about their stated law. Everything else is
counted as mechanism-specific (hand/cloud-written). See GitHub issue #268
and predictions/lean_stub_registration.json for the pre-registered claim
this census resolves.

    python3 morphoeval/census_lean_stubs.py            # writes morphoeval/problem_scouts/lean_stub_census.json
    python3 morphoeval/census_lean_stubs.py --check    # exit 1 if stub share changes (CI-style)
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOOPS = HERE / "lean" / "loops"
OUT = HERE / "problem_scouts" / "lean_stub_census.json"
STUB_LINE = "def law1_invariant (a b : Int) : Int := a * b"


def census() -> dict:
    stub, mechanism = [], []
    for f in sorted(LOOPS.glob("*.lean")):
        (stub if STUB_LINE in f.read_text(errors="replace") else mechanism).append(f.name)
    ns = {}
    for name in stub:
        m = re.match(r"(Loop_\d+)", name)
        if m:
            ns[m.group(1)] = name
    return {"stub_line": STUB_LINE, "total_lean_loop_files": len(stub) + len(mechanism),
            "stub_template_files": len(stub), "mechanism_specific_files": len(mechanism),
            "stub_namespaces": ns, "mechanism_files": mechanism}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    c = census()
    summary = {k: v for k, v in c.items() if k != "stub_namespaces"}
    if args.check:
        prev = json.loads(OUT.read_text()) if OUT.exists() else {}
        changed = prev.get("stub_template_files") != c["stub_template_files"]
        print(json.dumps(summary, indent=2))
        print("CHANGED" if changed else "unchanged")
        return 1 if changed else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
