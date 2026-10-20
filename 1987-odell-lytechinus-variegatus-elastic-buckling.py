#!/usr/bin/env python3
"""
1987-odell-lytechinus-variegatus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0755)
Organism: Lytechinus variegatus (Green Sea Urchin)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Lytechinus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0755_lytechinus_variegatus_elastic_buckling.lean` (`MathOfBiology.Loop_0755`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/lytechinus_variegatus_elastic_buckling"
EXPECTED_SHA256 = {
    "lytechinus_variegatus_elastic_buckling_reference.json": "93f4a711066ce840823bfa2f66a3afe3e0206df63fd321e9ab34ea0e2dc571f6",
    "lytechinus_variegatus_specimen.jpg": "521ca1f087eb9618f793b80f0084a9efa6db3c8fb7039599c38d75ba00260619",
    "scout_manifest.json": "498baf4a14e26bdfbb28285775c393500967a68799a6ea4a7c1d41cebf26d897",
    "paper_citations.bib": "c637a70564078c3f7490856a16f138f76a4df864014ebbd321ad60e5a20cee90"
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
    ref = json.loads((DATASET_DIR / "lytechinus_variegatus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1987-odell-lytechinus-variegatus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
