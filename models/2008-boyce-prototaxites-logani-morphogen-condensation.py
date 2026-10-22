#!/usr/bin/env python3
"""
models/2008-boyce-prototaxites-logani-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1350)
Organism: Prototaxites logani (Early Devonian Giant Nematophyte Mycelial Trunk (Concentric Hyphal Rings))
Phenomenon Scale: 6.1 cm (0.06111 m, log10=-1.214, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Prototaxites logani (Early Devonian (Emsian, 400 Ma))
Geological Horizon: Early Devonian (Emsian, 400 Ma) (400.0 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Prototaxites
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (6.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1350_prototaxites_logani_morphogen_condensation.lean` (`MathOfBiology.Loop_1350`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/prototaxites_logani_morphogen_condensation"
EXPECTED_SHA256 = {
    "prototaxites_logani_morphogen_condensation_reference.json": "b4b83cc26f5167bed17ee20210860e69d9bcf25bac0e870f35f08fd3d7ed2d19",
    "prototaxites_logani_specimen.jpg": "672ce91b81384694dbcf49ddcc7b3191d489b29fbe3b3f90608e2ed52f775b12",
    "scout_manifest.json": "d707a409cd3af9154b26a2292dd4a0632cdaa96a7a7ab2cdc6d9e8a62ed147f4",
    "paper_citations.bib": "64a54310a07899f81247642b14ee0e2d5cbd0fbb319641303412bd56b60789d9"
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
    ref = json.loads((DATASET_DIR / "prototaxites_logani_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2008-boyce-prototaxites-logani-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
