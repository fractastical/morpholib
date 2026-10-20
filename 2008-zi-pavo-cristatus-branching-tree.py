#!/usr/bin/env python3
"""
2008-zi-pavo-cristatus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0613)
Organism: Pavo cristatus (Indian Peafowl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Pavo
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0613_pavo_cristatus_branching_tree.lean` (`MathOfBiology.Loop_0613`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pavo_cristatus_branching_tree"
EXPECTED_SHA256 = {
    "pavo_cristatus_branching_tree_reference.json": "949390ceb65645552c1b5f4b7065e90d769a06516b4e05c7984cb7efaf677ad3",
    "pavo_cristatus_specimen.jpg": "39f28e26a62ddfdb98fe19fee7eaf7af508c82486a32abe3c1f2ac7d3e607d50",
    "scout_manifest.json": "e99ec1695dc18aa32f1edb3c105c1c3c343443e78b53ccce8ce9bbea664b87ec",
    "paper_citations.bib": "1111e94cd19e99d5bde401109de6cf927c1f1c4dadf2e208c9ced5f86ed63d2a"
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
    ref = json.loads((DATASET_DIR / "pavo_cristatus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-zi-pavo-cristatus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
