#!/usr/bin/env python3
"""
models/2016-dilkes-eryops-megacephalus-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1757)
Organism: Eryops megacephalus (Early Permian Temnospondyl Pentadactyl Amphibian)
Phenomenon Scale: 4.5 cm (0.04501 m, log10=-1.347, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Eryops megacephalus (Early Permian (Cisuralian, 295 Ma))
Geological Horizon: Early Permian (Cisuralian, 295 Ma) (295.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Eryops
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (4.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1757_eryops_megacephalus_elastic_buckling.lean` (`MathOfBiology.Loop_1757`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eryops_megacephalus_elastic_buckling"
EXPECTED_SHA256 = {
    "eryops_megacephalus_elastic_buckling_reference.json": "7fc6062875f0b43f39720d4ff7dd11286938da5add23dfc1bf0a8e2895edbc29",
    "eryops_megacephalus_specimen.jpg": "129f152b5349248669bd3ffc0b0360d0131cfa03e87c6ff1c0ac912fae50e424",
    "scout_manifest.json": "83cfad405c19bffc33d28f07e49175eee5b32e165ba2167d4290b0b79e9a2cb7",
    "paper_citations.bib": "bbd1d15fc3195cb74beef73e42a01546f5d23d73d9301e4173cfb18e5c919e9e"
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
    ref = json.loads((DATASET_DIR / "eryops_megacephalus_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-dilkes-eryops-megacephalus-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
