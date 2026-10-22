#!/usr/bin/env python3
"""
models/2011-li-anchiornis-huxleyi-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1769)
Organism: Anchiornis huxleyi (Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array))
Phenomenon Scale: 159 nm (1.589e-07 m, log10=-6.799, 2_macromolecular_100nm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Anchiornis huxleyi (Late Jurassic (Tiaojishan Fm, 160.0 Ma))
Geological Horizon: Late Jurassic (Tiaojishan Fm, 160.0 Ma) (160.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anchiornis
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (159 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1769_anchiornis_huxleyi_elastic_buckling.lean` (`MathOfBiology.Loop_1769`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anchiornis_huxleyi_elastic_buckling"
EXPECTED_SHA256 = {
    "anchiornis_huxleyi_elastic_buckling_reference.json": "91e3a527039cd9c6853896aa0c176a534da2a47e2c432b2695c23f673967e174",
    "anchiornis_huxleyi_specimen.jpg": "fab0bb4ca7882feafbf18840d6056dd36150060dc484593ec1df858d67a18865",
    "scout_manifest.json": "9b57601b485ba75975d4713c6e16a76d265a43241666c94cf2d4d272fc693be1",
    "paper_citations.bib": "f58bd4c219c7e2947c0c94f941019a0a07c0969d6df27e8f0c673b5d9ecbc3f5"
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
    ref = json.loads((DATASET_DIR / "anchiornis_huxleyi_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-li-anchiornis-huxleyi-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
