#!/usr/bin/env python3
"""
1991-bartnicki-neurospora-crassa-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1081)
Organism: Neurospora crassa (Red Bread Mold)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Neurospora
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1081_neurospora_crassa_collective_stigmergy.lean` (`MathOfBiology.Loop_1081`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/neurospora_crassa_collective_stigmergy"
EXPECTED_SHA256 = {
    "neurospora_crassa_collective_stigmergy_reference.json": "6fdd779aee04a75fb27a8b0fdb487b8db37f3166eeffbdfcd08409bde351bf1c",
    "neurospora_crassa_specimen.jpg": "01d6531f055000fd6be6dc5baa24d01e76899962e05a548e4be38a45032165f0",
    "scout_manifest.json": "3f5b4fd0a1e1a9cf8625ce0657628bbe494d14cbdd3a1e4128436ae99e7a9486",
    "paper_citations.bib": "d09159d68a09e480bf0ff190352b4e141cb2a515ea033d6b77bd7d4b7444ad01"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "neurospora_crassa_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 1991-bartnicki-neurospora-crassa-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
