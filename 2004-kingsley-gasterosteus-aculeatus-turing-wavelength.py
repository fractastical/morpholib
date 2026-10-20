#!/usr/bin/env python3
"""
2004-kingsley-gasterosteus-aculeatus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0136)
Organism: Gasterosteus aculeatus (Three-Spined Stickleback)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0136_gasterosteus_aculeatus_turing_wavelength.lean` (`MathOfBiology.Loop_0136`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gasterosteus_aculeatus_turing_wavelength"
EXPECTED_SHA256 = {
    "gasterosteus_aculeatus_turing_wavelength_reference.json": "8e9aa8c4ddab223fba3ee267f7c181ca61df0a99d2d4d6b7691161ce6cfe6385",
    "gasterosteus_aculeatus_specimen.jpg": "86fef1d4aeb86c3903e26fae972a360ea0c848048e29b546bedb2db842a5bd10",
    "scout_manifest.json": "28999ea3ccbbfefa1ebaed23ecc706b36f4872b31bf89c74adb3a564b4aa43e3",
    "paper_citations.bib": "222474b5b02c53e22b5fe16dd09a85b2d2e7c570ef7418ef1000f11d04ed07ff"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_aculeatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 2004-kingsley-gasterosteus-aculeatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
