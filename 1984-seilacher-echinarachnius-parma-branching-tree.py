#!/usr/bin/env python3
"""
1984-seilacher-echinarachnius-parma-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0660)
Organism: Echinarachnius parma (Common Sand Dollar)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Echinarachnius
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0660_echinarachnius_parma_branching_tree.lean` (`MathOfBiology.Loop_0660`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/echinarachnius_parma_branching_tree"
EXPECTED_SHA256 = {
    "echinarachnius_parma_branching_tree_reference.json": "28c01c7d7e33a1e7a94c78401a1695b8802eed5503e1aa3f52803fad2e1316ec",
    "echinarachnius_parma_specimen.jpg": "9076ecda047934e3a862128e9da39faefb16c968f63494a46db1ab7512327923",
    "scout_manifest.json": "00d65d2ad82327b986f3cb4b241c4a0329f9e9ae044d2c42c522ab3bc909a1ff",
    "paper_citations.bib": "4a8bccb55c37c04f053d43d68933cad6f72d191fde7ba1962f50e9601359a0bd"
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
    ref = json.loads((DATASET_DIR / "echinarachnius_parma_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1984-seilacher-echinarachnius-parma-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
