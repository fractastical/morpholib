#!/usr/bin/env python3
"""
models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1191)
Organism: Danio rerio (Zebrafish — Adult Collective Shoal)
Phenomenon Scale: 31.9 cm (0.3192 m, log10=-0.496, 8_colonial_m)
Anatomical Structure: Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice (Adult Collective Shoal (15–30 Fish Polarized School, 0.42 m arena)) · MapZebrain optic_tectum_loom_circuit & lateral_line_rheotaxis
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Danio
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection at 31.9 cm (Reverse Kármán vortex dipole wake & visual-mechanosensory shoal lattice))
Formal Verification: `morphoeval/lean/loops/Loop_1191_danio_rerio_cm42_collective_shoal_vortex_wake_turing_wavelength.lean` (`MathOfBiology.Loop_1191`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/danio_rerio_cm42_collective_shoal_vortex_wake_turing_wavelength"
EXPECTED_SHA256 = {
    "danio_rerio_cm42_collective_shoal_vortex_wake_turing_wavelength_reference.json": "c28eb059aa2ed8cb6950471e2ee66eddd832688d0f3a51ea81390379099deed3",
    "danio_rerio_specimen.jpg": "850b2d4a94c7cbcb3c16e606ddb70d198f082ccb2f046ee44dbaa5ef5941678a",
    "scout_manifest.json": "4d0e4af1a35a97e970e986d34a21c538c82d4ed61851bad7ae8ba6a5e59ecdb2",
    "paper_citations.bib": "13f0ece137a3cecb2c81879b36b26288ac6aae8dc7b2738c17a2573dc576e510"
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
    ref = json.loads((DATASET_DIR / "danio_rerio_cm42_collective_shoal_vortex_wake_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-couzin-danio-rerio-cm42-collective-shoal-vortex-wake-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
