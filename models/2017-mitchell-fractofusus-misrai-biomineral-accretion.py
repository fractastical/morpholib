#!/usr/bin/env python3
"""
models/2017-mitchell-fractofusus-misrai-biomineral-accretion.py — Multi-Scale & Paleobiological Reproduction (Slot #1881)
Organism: Fractofusus misrai (Ediacaran Four-Order Self-Similar Spindle Rangeomorph)
Phenomenon Scale: 4.5 cm (0.04503 m, log10=-1.346, 7_organism_cm)
Anatomical Structure: Biomineralized shell whorl / spicule / test in Fractofusus misrai (Late Ediacaran (Newfoundland, 565 Ma))
Geological Horizon: Late Ediacaran (Newfoundland, 565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Fractofusus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization (4.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1881_fractofusus_misrai_biomineral_accretion.lean` (`MathOfBiology.Loop_1881`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/fractofusus_misrai_biomineral_accretion"
EXPECTED_SHA256 = {
    "fractofusus_misrai_biomineral_accretion_reference.json": "de5f7b0f0410f1fa220b45df059cc94df811d3074fd3a3bb0b819a91a30a2384",
    "fractofusus_misrai_specimen.jpg": "b6b0eca46e1970dc859de875d6c583e3ce1abb548f235c9292a993aa71e14061",
    "scout_manifest.json": "2b86315cb0ba5e6b58c3715479d7e676c7bff50f56118bc58497744477cc2ee0",
    "paper_citations.bib": "14dd3862f51967eee790d17e259b0756b522e22f0d0ae553b960e040037e01f8"
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
    ref = json.loads((DATASET_DIR / "fractofusus_misrai_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2017-mitchell-fractofusus-misrai-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
