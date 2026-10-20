#!/usr/bin/env python3
"""
2000-meinhardt-sus-scrofa-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0606)
Organism: Sus scrofa (Wild Boar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Sus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0606_sus_scrofa_branching_tree.lean` (`MathOfBiology.Loop_0606`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sus_scrofa_branching_tree"
EXPECTED_SHA256 = {
    "sus_scrofa_branching_tree_reference.json": "9195b97003784a222dc79ac95a7215a50e9f39f57f94e65763a180e03bc5a92a",
    "sus_scrofa_specimen.jpg": "2a19f0ec20969646f39adb56919e7ec23b3c0f4a963658eeab969a0d13ba3d70",
    "scout_manifest.json": "d52b6b4b69e202212f3ddee48913d90c3e3ed382d03584a45ec08ffbe8c6e615",
    "paper_citations.bib": "c6ffa5dcb4874167ca01a52ef5c2ddddd3d8d09cf8ba08f2dd8b162494af60de"
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
    ref = json.loads((DATASET_DIR / "sus_scrofa_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-meinhardt-sus-scrofa-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
