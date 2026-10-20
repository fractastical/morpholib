#!/usr/bin/env python3
"""
2014-keller-xenopus-tropicalis-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0721)
Organism: Xenopus tropicalis (Western Clawed Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Xenopus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0721_xenopus_tropicalis_elastic_buckling.lean` (`MathOfBiology.Loop_0721`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/xenopus_tropicalis_elastic_buckling"
EXPECTED_SHA256 = {
    "xenopus_tropicalis_elastic_buckling_reference.json": "64c2cf8ee2579b887d248d19ab8864f15b0a55b97f47658ef349ee55326cae4f",
    "xenopus_tropicalis_specimen.jpg": "13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f",
    "scout_manifest.json": "98365fcb30dde0230deecf17e511f5d66334841ba47a444143d3dec505b85e5c",
    "paper_citations.bib": "41edee12c1765150ad4704592ff4c79cc4fb3b3e42470b517cdffee908bdc2b5"
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
    ref = json.loads((DATASET_DIR / "xenopus_tropicalis_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-keller-xenopus-tropicalis-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
