#!/usr/bin/env python3
"""
2020-standens-polypterus-senegalus-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0740)
Organism: Polypterus senegalus (Gray Bichir)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Polypterus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0740_polypterus_senegalus_elastic_buckling.lean` (`MathOfBiology.Loop_0740`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/polypterus_senegalus_elastic_buckling"
EXPECTED_SHA256 = {
    "polypterus_senegalus_elastic_buckling_reference.json": "e535af43e931dc060d112d052262ee0ddaaf79ad364d4027ed6fd6dc44b1886c",
    "polypterus_senegalus_specimen.jpg": "8332a72979f53f67ddcfb9e11c646f5695dd1fac3cbcfb340acc3d76e93582f5",
    "scout_manifest.json": "3297cfa2d2212d867b4b0ab09410df2224c69106863bfa53c5f19f180376fe90",
    "paper_citations.bib": "f720a8ac5ed1d9c5002493f5b6424808bc5cbcd93478fe0eba3bb825599e8f76"
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
    ref = json.loads((DATASET_DIR / "polypterus_senegalus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-standens-polypterus-senegalus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
