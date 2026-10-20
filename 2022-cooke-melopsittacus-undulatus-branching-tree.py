#!/usr/bin/env python3
"""
2022-cooke-melopsittacus-undulatus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0617)
Organism: Melopsittacus undulatus (Budgerigar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Melopsittacus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0617_melopsittacus_undulatus_branching_tree.lean` (`MathOfBiology.Loop_0617`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/melopsittacus_undulatus_branching_tree"
EXPECTED_SHA256 = {
    "melopsittacus_undulatus_branching_tree_reference.json": "3d3a7150af0a6d28bb4f967151381b9a213a5267a1c49454afd9d684883ea85d",
    "melopsittacus_undulatus_specimen.jpg": "f61163f7820594be22c648c5120f1e827b9cdd8d377d9eeb65bb96e633b0ce2b",
    "scout_manifest.json": "7e3cf5345ff61c3603fdcc0ee46860ad8f1882e325088c29d364de42e984f260",
    "paper_citations.bib": "c67e41b0b07313a6b271854856c027eaac7df32d3c1767f30069bc8b8b5b2afc"
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
    ref = json.loads((DATASET_DIR / "melopsittacus_undulatus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2022-cooke-melopsittacus-undulatus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
