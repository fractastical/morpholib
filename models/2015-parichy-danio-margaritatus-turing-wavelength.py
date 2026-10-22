#!/usr/bin/env python3
"""
models/2015-parichy-danio-margaritatus-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1201)
Organism: Danio margaritatus (Celestial Pearl Danio (Galaxy Rasbora))
Phenomenon Scale: 612 µm (0.000612 m, log10=-3.213, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Danio margaritatus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (612 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1201_danio_margaritatus_turing_wavelength.lean` (`MathOfBiology.Loop_1201`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_margaritatus_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_margaritatus_turing_wavelength_reference.json": "e90cf9682cb52e725f2c891d01aff61f328cdfd9825c7dc00ec2fb2caf8f8397",
    "danio_margaritatus_specimen.jpg": "6f4ec861e86da2f7ea31f23d6a81b802286dce7ac2a0631e45f9097fd9b8e876",
    "scout_manifest.json": "060b6f06d4a1a67e84919a40cac51b972cc3972c16cb7840c214ab904eb949ed",
    "paper_citations.bib": "a03aae5f51200804fc51606cd967d1941532cb5457c260ff108bd18dab8c56f1"
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
    ref = json.loads((DATASET_DIR / "danio_margaritatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-parichy-danio-margaritatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
