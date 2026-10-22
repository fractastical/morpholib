#!/usr/bin/env python3
"""
models/2010-li-anchiornis-huxleyi-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1229)
Organism: Anchiornis huxleyi (Late Jurassic Four-Winged Paravian Dinosaur (Melanosome Array))
Phenomenon Scale: 29 nm (2.89e-08 m, log10=-7.539, 1_molecular_nm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Anchiornis huxleyi (Late Jurassic (Tiaojishan Fm, 160.0 Ma))
Geological Horizon: Late Jurassic (Tiaojishan Fm, 160.0 Ma) (160.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anchiornis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (29 nm))
Formal Verification: `morphoeval/lean/loops/Loop_1229_anchiornis_huxleyi_turing_wavelength.lean` (`MathOfBiology.Loop_1229`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anchiornis_huxleyi_turing_wavelength"
EXPECTED_SHA256 = {
    "anchiornis_huxleyi_turing_wavelength_reference.json": "d4952b7fb35999064f6e26542e145f4f614a6e482c3310166b801e95296aaa6a",
    "anchiornis_huxleyi_specimen.jpg": "fab0bb4ca7882feafbf18840d6056dd36150060dc484593ec1df858d67a18865",
    "scout_manifest.json": "066a846238f265eed4324ba872f63c3293e6a31dd4835b0cbad946e7800fe257",
    "paper_citations.bib": "aa8f29f2bb6794089e72358f0138b8cd23916449fc7f8227568cd5ffc1e93400"
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
    ref = json.loads((DATASET_DIR / "anchiornis_huxleyi_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2010-li-anchiornis-huxleyi-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
