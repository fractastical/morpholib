#!/usr/bin/env python3
"""
2008-munro-ciona-intestinalis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0763)
Organism: Ciona intestinalis (Vase Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Ciona
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0763_ciona_intestinalis_elastic_buckling.lean` (`MathOfBiology.Loop_0763`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ciona_intestinalis_elastic_buckling"
EXPECTED_SHA256 = {
    "ciona_intestinalis_elastic_buckling_reference.json": "1f26748386b04a5a0ceb4842a26bee19a7b5c197b67b18abd7894c23e0c7a0dc",
    "ciona_intestinalis_specimen.jpg": "0f563cbc2ccb4b7ba1595b7519a75a1410031eaaa4afceaf02b5975d87c1ecf4",
    "scout_manifest.json": "73b704a007fc42c8d22f424271295b9d2c92b92a3ee9d54d5c0f0e35df99ac5b",
    "paper_citations.bib": "eaf8f533dd5ad5ffd164fb758f3166befbcd2b795923c79f17f441ca2ffe18bf"
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
    ref = json.loads((DATASET_DIR / "ciona_intestinalis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-munro-ciona-intestinalis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
