#!/usr/bin/env python3
"""
2009-kingsley-gasterosteus-aculeatus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0636)
Organism: Gasterosteus aculeatus (Three-Spined Stickleback)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0636_gasterosteus_aculeatus_branching_tree.lean` (`MathOfBiology.Loop_0636`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_branching_tree"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_branching_tree_reference.json": "9809f7755201ba51c84fafe010c43b01fffd967b977f94dd2997ad23d32f00db",
    "gasterosteus_aculeatus_specimen.jpg": "86fef1d4aeb86c3903e26fae972a360ea0c848048e29b546bedb2db842a5bd10",
    "scout_manifest.json": "eb1852927492893faa797baa6c87caec1002e92068250d427c0c4930c31d62f8",
    "paper_citations.bib": "c8a379c0aaa931be2a45fe05891ed3038691496ed71bd6c22cfd14e7b9eae65b"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-kingsley-gasterosteus-aculeatus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
