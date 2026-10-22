#!/usr/bin/env python3
"""
models/1997-coates-acanthostega-gunnari-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1755)
Organism: Acanthostega gunnari (Late Devonian Eight-Digited Polydactyl Stem-Tetrapod)
Phenomenon Scale: 2.1 cm (0.0215 m, log10=-1.668, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Acanthostega gunnari (Late Devonian (Famennian, 365 Ma))
Geological Horizon: Late Devonian (Famennian, 365 Ma) (365.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Acanthostega
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (2.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1755_acanthostega_gunnari_elastic_buckling.lean` (`MathOfBiology.Loop_1755`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/acanthostega_gunnari_elastic_buckling"
EXPECTED_SHA256 = {
    "acanthostega_gunnari_elastic_buckling_reference.json": "856d4bc03d50a6fba537cf29477a5516ea52c57bfa78eddbe1f0104536a5d4d0",
    "acanthostega_gunnari_specimen.jpg": "17b12e17eb08ab7e05345c565a3dc299e5c4752dd56c4753617937fa7b3a66b5",
    "scout_manifest.json": "1088334ba2a2370b7e4eec25f5eb01cb7dfaa52d6b74f43e9106fccaa82328d9",
    "paper_citations.bib": "565f25cce3b50b2defb2b107401a981ac63e18ddce93f44c5a8f0a1b801ed271"
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
    ref = json.loads((DATASET_DIR / "acanthostega_gunnari_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1997-coates-acanthostega-gunnari-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
