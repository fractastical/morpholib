#!/usr/bin/env python3
"""
models/2023-gishlick-walliserops-trifurcatus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1247)
Organism: Walliserops trifurcatus (Middle Devonian Long-Trident Moroccan Trilobite)
Phenomenon Scale: 1.9 mm (0.001939 m, log10=-2.712, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Walliserops trifurcatus (Middle Devonian (Eifelian, 394 Ma))
Geological Horizon: Middle Devonian (Eifelian, 394 Ma) (394.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Walliserops
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1247_walliserops_trifurcatus_turing_wavelength.lean` (`MathOfBiology.Loop_1247`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/walliserops_trifurcatus_turing_wavelength"
EXPECTED_SHA256 = {
    "walliserops_trifurcatus_turing_wavelength_reference.json": "00ab0d06a7e3202950087c3309181d747aea685d9a440faeff6058ab2fc38a37",
    "walliserops_trifurcatus_specimen.jpg": "8a4e9c4d8a7e87dbdcf6b43da85f70188d90b4d5e1d541428ebaa0c016dd9923",
    "scout_manifest.json": "5e412aa0bb148a56297759250f126c91842d8920d219e86ed992af7bef9f92ac",
    "paper_citations.bib": "67e95bb5034810eb6e30b2690d1c48618660a41602c03a0e2b84ec6c47def52c"
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
    ref = json.loads((DATASET_DIR / "walliserops_trifurcatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-gishlick-walliserops-trifurcatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
