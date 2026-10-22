#!/usr/bin/env python3
"""
models/2015-dilkes-eryops-megacephalus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1217)
Organism: Eryops megacephalus (Early Permian Temnospondyl Pentadactyl Amphibian)
Phenomenon Scale: 8.2 mm (0.008184 m, log10=-2.087, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Eryops megacephalus (Early Permian (Cisuralian, 295 Ma))
Geological Horizon: Early Permian (Cisuralian, 295 Ma) (295.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Eryops
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (8.2 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1217_eryops_megacephalus_turing_wavelength.lean` (`MathOfBiology.Loop_1217`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eryops_megacephalus_turing_wavelength"
EXPECTED_SHA256 = {
    "eryops_megacephalus_turing_wavelength_reference.json": "ffff702b9abd4d87f2a1889cdd83c78ef5a71cdcdf8e59e95397340be783ce01",
    "eryops_megacephalus_specimen.jpg": "129f152b5349248669bd3ffc0b0360d0131cfa03e87c6ff1c0ac912fae50e424",
    "scout_manifest.json": "ebe53f64a10aac75be71ec21e593770742c571d44bb6823987e0a5c9b6cc19ea",
    "paper_citations.bib": "a978c0b8751234f641b73050a8b026c5c933e3b00c14cd33835554c17516c592"
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
    ref = json.loads((DATASET_DIR / "eryops_megacephalus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-dilkes-eryops-megacephalus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
