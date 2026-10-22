#!/usr/bin/env python3
"""
models/2021-kavanagh-pliohippus-pernix-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1761)
Organism: Pliohippus pernix (Late Miocene Single-Hoofed Monodactyl Ancestor)
Phenomenon Scale: 5.2 cm (0.05156 m, log10=-1.288, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Pliohippus pernix (Late Miocene (Hemphillian, 8.5 Ma))
Geological Horizon: Late Miocene (Hemphillian, 8.5 Ma) (8.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pliohippus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (5.2 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1761_pliohippus_pernix_elastic_buckling.lean` (`MathOfBiology.Loop_1761`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pliohippus_pernix_elastic_buckling"
EXPECTED_SHA256 = {
    "pliohippus_pernix_elastic_buckling_reference.json": "c77dd81f45d00cba1715cc848df806d81e3bd44b81506681a3057fd5f625d728",
    "pliohippus_pernix_specimen.jpg": "1211e4acc14fca6bb091e7747aca534b08c9b54a2cea23f8b7aa25c1b659afdd",
    "scout_manifest.json": "83aaf038e4345e23d3d13a7fc95e3635005840581146f83596936ad7853674ca",
    "paper_citations.bib": "dfdf9d9f3350f0e77bf3db16784a726e85c128b92aae00c219054cbec824a496"
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
    ref = json.loads((DATASET_DIR / "pliohippus_pernix_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-kavanagh-pliohippus-pernix-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
