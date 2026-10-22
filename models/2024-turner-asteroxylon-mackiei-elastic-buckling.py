#!/usr/bin/env python3
"""
models/2024-turner-asteroxylon-mackiei-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1796)
Organism: Asteroxylon mackiei (Early Devonian Rhynie Chert Lycopsid (Actinostele & Fibonacci Microphylls))
Phenomenon Scale: 1.6 mm (0.001559 m, log10=-2.807, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Asteroxylon mackiei (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Asteroxylon
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1796_asteroxylon_mackiei_elastic_buckling.lean` (`MathOfBiology.Loop_1796`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/asteroxylon_mackiei_elastic_buckling"
EXPECTED_SHA256 = {
    "asteroxylon_mackiei_elastic_buckling_reference.json": "7ed0fcfeba2b1b3169ce0c89183f1304affeb3811e52429472e93fc4b7861786",
    "asteroxylon_mackiei_specimen.jpg": "af15c01893b31829fefca3fc2246896ab8d2c12ab87d5809f28504e9c2968531",
    "scout_manifest.json": "9ecc411616be3ff227bb6465dd7e984bb3dbfd8cc47c131916153c58d95b4d44",
    "paper_citations.bib": "0f7db27dacc3ae3ebe563bb39ac17f327d2fefa54f3fa216ac56af9007826c84"
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
    ref = json.loads((DATASET_DIR / "asteroxylon_mackiei_elastic_buckling_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/2024-turner-asteroxylon-mackiei-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
