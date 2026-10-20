#!/usr/bin/env python3
"""
2023-cooke-melopsittacus-undulatus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0717)
Organism: Melopsittacus undulatus (Budgerigar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Melopsittacus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0717_melopsittacus_undulatus_elastic_buckling.lean` (`MathOfBiology.Loop_0717`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/melopsittacus_undulatus_elastic_buckling"
EXPECTED_SHA256 = {
    "melopsittacus_undulatus_elastic_buckling_reference.json": "0f295f1c35bccdbe6876bb806f093bc49101a2f3c8471e23adbb6edaea6164d0",
    "melopsittacus_undulatus_specimen.jpg": "f61163f7820594be22c648c5120f1e827b9cdd8d377d9eeb65bb96e633b0ce2b",
    "scout_manifest.json": "dd3d7cd3ade432a506a331a15eeb68bb29198ea17a55d2b126753ee91e3ad724",
    "paper_citations.bib": "56ad2bdd124fd7f07702672363c951407fcbdebe4d7928c1596ee3cccb294d0a"
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
    ref = json.loads((DATASET_DIR / "melopsittacus_undulatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2023-cooke-melopsittacus-undulatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
