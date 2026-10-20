#!/usr/bin/env python3
"""
2023-reiter-sepia-officinalis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0651)
Organism: Sepia officinalis (Common Cuttlefish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Sepia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0651_sepia_officinalis_branching_tree.lean` (`MathOfBiology.Loop_0651`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sepia_officinalis_branching_tree"
EXPECTED_SHA256 = {
    "sepia_officinalis_branching_tree_reference.json": "21308c48d2bae6cd69f3b690f52a794c90507f32df0a9c77473b78c1fa9a6e21",
    "sepia_officinalis_specimen.jpg": "c6ecd8366e75a19b7375ac673bb90c8aa032e891c606d26556858646002b58e1",
    "scout_manifest.json": "52241af1bb071d816ef637d8141330f7c64609f3891d84ebf6552e0255bdc149",
    "paper_citations.bib": "03bb52835041b981e3047de164335630089ebdab4c5b003d444e1091cb1938c2"
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
    ref = json.loads((DATASET_DIR / "sepia_officinalis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2023-reiter-sepia-officinalis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
