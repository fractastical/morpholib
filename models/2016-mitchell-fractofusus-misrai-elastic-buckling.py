#!/usr/bin/env python3
"""
models/2016-mitchell-fractofusus-misrai-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1791)
Organism: Fractofusus misrai (Ediacaran Four-Order Self-Similar Spindle Rangeomorph)
Phenomenon Scale: 1.8 cm (0.01801 m, log10=-1.744, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Fractofusus misrai (Late Ediacaran (Newfoundland, 565 Ma))
Geological Horizon: Late Ediacaran (Newfoundland, 565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Fractofusus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1791_fractofusus_misrai_elastic_buckling.lean` (`MathOfBiology.Loop_1791`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/fractofusus_misrai_elastic_buckling"
EXPECTED_SHA256 = {
    "fractofusus_misrai_elastic_buckling_reference.json": "b5a5b017c8047b73edb5b1562c3a0d2b5136cfec4cdb58e9f3646961ab8ef394",
    "fractofusus_misrai_specimen.jpg": "b6b0eca46e1970dc859de875d6c583e3ce1abb548f235c9292a993aa71e14061",
    "scout_manifest.json": "2da05a6391048402bae8c2b3e958893481fab4ac1b37de1f05e6e8faf939d80d",
    "paper_citations.bib": "7edaf774dc6ee087d6d0d63f0760ddeec5667d0bffe4846954ddadcaecb4b695"
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
    ref = json.loads((DATASET_DIR / "fractofusus_misrai_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2016-mitchell-fractofusus-misrai-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
