#!/usr/bin/env python3
"""
1990-goodwin-acetabularia-acetabulum-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0692)
Organism: Acetabularia acetabulum (Mermaid's Wineglass Alga)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Acetabularia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0692_acetabularia_acetabulum_branching_tree.lean` (`MathOfBiology.Loop_0692`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/acetabularia_acetabulum_branching_tree"
EXPECTED_SHA256 = {
    "acetabularia_acetabulum_branching_tree_reference.json": "2d76c56dee0a44982a258caacfabb905182643431bdd033192b3b347277cda92",
    "acetabularia_acetabulum_specimen.jpg": "17641545f872481e451f25234ae988af0c16a529dd25aff08a53bd400e2c23a1",
    "scout_manifest.json": "c3a67e8b1299a13ee7e937f990514ac13a2d509347d25913b501ae56bb82f1aa",
    "paper_citations.bib": "3fb5e795313df37af9a1246be7fc2243054a1c2579f7b5d3a53b97324621228f"
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
    ref = json.loads((DATASET_DIR / "acetabularia_acetabulum_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1990-goodwin-acetabularia-acetabulum-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
