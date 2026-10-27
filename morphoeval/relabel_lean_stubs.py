#!/usr/bin/env python3
"""Relabel overnight-template Lean stubs so nothing downstream overclaims.

Two idempotent passes (see census_lean_stubs.py, GitHub issue #268):
1. Every stub file in morphoeval/lean/loops/ gets an explicit
   `STATUS: PLACEHOLDER TEMPLATE` line inside its header comment.
2. Every morphoeval/math_kg_loops/*.yaml whose equations point at a stub
   namespace (`lean_decl: MathOfBiology.Loop_<N>.law*_…`) gets a top-level
   `lean_verification: placeholder_not_mechanism_proof` marker, so the KG
   no longer asserts that the named law is formally verified.

Genuine (non-stub) files are never touched. Safe to re-run.

    python3 morphoeval/relabel_lean_stubs.py
"""
from __future__ import annotations
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE / "problem_scouts" / "lean_stub_census.json"
LOOPS = HERE / "lean" / "loops"
KG = HERE / "math_kg_loops"
MARK = "STATUS: PLACEHOLDER TEMPLATE — arithmetic sanity lemmas only (a*b positivity, p+q-q=p, constant threshold by decide). NOT a formalisation of the law named above; see census morphoeval/problem_scouts/lean_stub_census.json and issue #268."


def main() -> int:
    c = json.loads(CENSUS.read_text())
    stub_ns = set(c["stub_namespaces"])
    n_files = 0
    for name in c["stub_namespaces"].values():
        f = LOOPS / name
        text = f.read_text(errors="replace")
        if MARK in text:
            continue
        lines = text.splitlines(keepends=True)
        # insert inside the leading /- ... -/ header, right before its closing line
        for i, ln in enumerate(lines):
            if ln.strip() == "-/":
                lines.insert(i, f"  {MARK}\n")
                break
        else:
            lines.insert(0, f"-- {MARK}\n")
        f.write_text("".join(lines))
        n_files += 1
    n_yaml = 0
    for y in sorted(KG.glob("*.yaml")):
        text = y.read_text(errors="replace")
        if "lean_verification:" in text:
            continue
        refs = set(re.findall(r"MathOfBiology\.(Loop_\d+)\.", text))
        if refs & stub_ns:
            y.write_text("lean_verification: placeholder_not_mechanism_proof  # stub-linked laws; see issue #268\n" + text)
            n_yaml += 1
    print(f"Relabelled {n_files} stub Lean files, {n_yaml} KG YAMLs (idempotent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
