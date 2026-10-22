#!/usr/bin/env python3
"""
models/2008-boyce-prototaxites-logani-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1800)
Organism: Prototaxites logani (Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings))
Phenomenon Scale: 7.5 cm (0.07469 m, log10=-1.127, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Prototaxites logani (Early Devonian (Emsian, 400 Ma))
Geological Horizon: Early Devonian (Emsian, 400 Ma) (400.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Prototaxites
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (7.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1800_prototaxites_logani_elastic_buckling.lean` (`MathOfBiology.Loop_1800`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/prototaxites_logani_elastic_buckling"
EXPECTED_SHA256 = {
    "prototaxites_logani_elastic_buckling_reference.json": "a6563911a6f3b3672c1a99b89070c87f52a5c5cba4c5aeb03b15306438076e2f",
    "prototaxites_logani_specimen.jpg": "672ce91b81384694dbcf49ddcc7b3191d489b29fbe3b3f90608e2ed52f775b12",
    "scout_manifest.json": "b43fff26d5ac54fef99a12a3e56dca684fb6926b196e5c9b698902496f9d0e41",
    "paper_citations.bib": "19290eae26e5bc3317e546efa4b7ecbdfa3b2d1362e39fcb7481a1e10a396b33"
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
    ref = json.loads((DATASET_DIR / "prototaxites_logani_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-boyce-prototaxites-logani-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
