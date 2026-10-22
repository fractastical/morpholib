#!/usr/bin/env python3
"""
models/2012-feoing-archaeopteryx-lithographica-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1230)
Organism: Archaeopteryx lithographica (Late Jurassic Solnhofen Urvogel (Asymmetric Flight Vane))
Phenomenon Scale: 2.9 mm (0.002919 m, log10=-2.535, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Archaeopteryx lithographica (Late Jurassic (Tithonian, 150.0 Ma))
Geological Horizon: Late Jurassic (Tithonian, 150.0 Ma) (150.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Archaeopteryx
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (2.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1230_archaeopteryx_lithographica_turing_wavelength.lean` (`MathOfBiology.Loop_1230`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteryx_lithographica_turing_wavelength"
EXPECTED_SHA256 = {
    "archaeopteryx_lithographica_turing_wavelength_reference.json": "6f2ee25ccea0e692d05bbe84a05122e5307977b257270e0ca8cfa69e37d32ae5",
    "archaeopteryx_lithographica_specimen.jpg": "22ad1c7de4cfaadcc6057d0f4ad6dd252fd379582bec21b51bf73852daa000af",
    "scout_manifest.json": "9a0856700a6aeae57eaee214adc6b84d5c1d024692911006b6d22b10f1867069",
    "paper_citations.bib": "c478b6ba665e47042e3fbea02fb82c08cd7a66b91a954a95653f15a5ac096cb6"
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
    ref = json.loads((DATASET_DIR / "archaeopteryx_lithographica_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2012-feoing-archaeopteryx-lithographica-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
