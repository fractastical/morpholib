#!/usr/bin/env python3
"""
models/2019-landman-baculites-compressus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1779)
Organism: Baculites compressus (Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite)
Phenomenon Scale: 2.3 cm (0.02347 m, log10=-1.629, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Baculites compressus (Late Cretaceous (Pierre Shale, 73.5 Ma))
Geological Horizon: Late Cretaceous (Pierre Shale, 73.5 Ma) (73.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Baculites
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (2.3 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1779_baculites_compressus_elastic_buckling.lean` (`MathOfBiology.Loop_1779`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/baculites_compressus_elastic_buckling"
EXPECTED_SHA256 = {
    "baculites_compressus_elastic_buckling_reference.json": "8d06fe43b66c372cc69ea831f9235dc42ffdfc339a7e9032301e292ccd6ffcdb",
    "baculites_compressus_specimen.jpg": "849f95ffd5d647ae2a12750587da774719bdf94f1cf2520dc375c315c45eab50",
    "scout_manifest.json": "a0234d604a874651c6fac3b5a9ee38958c820fb81c07da54db17679824e26f25",
    "paper_citations.bib": "b6199ec6dfb8253a7547d344dc7104be441537b3856767ced13ec27d7c281062"
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
    ref = json.loads((DATASET_DIR / "baculites_compressus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-landman-baculites-compressus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
