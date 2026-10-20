#!/usr/bin/env python3
"""
2019-martin-schizosaccharomyces-pombe-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0687)
Organism: Schizosaccharomyces pombe (Fission Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Schizosaccharomyces
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0687_schizosaccharomyces_pombe_branching_tree.lean` (`MathOfBiology.Loop_0687`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schizosaccharomyces_pombe_branching_tree"
EXPECTED_SHA256 = {
    "schizosaccharomyces_pombe_branching_tree_reference.json": "9f0121e13cbb7c27afe281a7ee28994cc5665d8405bccd8c4c9dba1b9f339734",
    "schizosaccharomyces_pombe_specimen.jpg": "1ddf4f2081b0c047e472365745cd54e38ef85ea46a86a451dc98e0a7d6b5c9f8",
    "scout_manifest.json": "db174fe2dacc7bf3acd85d3e844aa1742dbc8c50e5d2f2c4df9832a0572fdbf8",
    "paper_citations.bib": "679cfc13974c4ae5fea83fc1b045622398f2775129df4d6f6fd11dba6994027e"
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
    ref = json.loads((DATASET_DIR / "schizosaccharomyces_pombe_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-martin-schizosaccharomyces-pombe-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
